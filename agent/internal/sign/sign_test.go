package sign

import (
	"bytes"
	"crypto/ed25519"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"github.com/anushka11p/Assay-health-grades-for-GPU-backed-debt/agent/internal/report"
)

// testKey is a fixed key so tests are repeatable. Never use a fixed key for real.
func testKey(seed byte) ed25519.PrivateKey {
	s := make([]byte, ed25519.SeedSize)
	s[0] = seed
	return ed25519.NewKeyFromSeed(s)
}

func sampleReport() report.Report {
	return report.Report{
		SchemaVersion: 1,
		ReportID:      "r1",
		CreatedAt:     "2026-11-09T12:00:00Z",
		Agent:         report.Agent{Version: "test", Backend: "fake"},
		GPUs: []report.GPU{{
			Identity: report.Identity{UUID: "GPU-1", Serial: "S1"},
			Memory:   report.MemoryHealth{Aggregate: report.ECCCounts{Uncorrected: report.Ptr(uint64(0))}},
			Samples:  []report.Sample{{TempC: report.Ptr(int64(60))}},
			Events:   []report.XIDEvent{},
		}},
	}
}

func TestSignThenVerify(t *testing.T) {
	r := sampleReport()
	if err := Sign(&r, testKey(1)); err != nil {
		t.Fatal(err)
	}
	res, err := Verify(r, nil)
	if err != nil {
		t.Fatal(err)
	}
	if res.Pinned || res.KeyID == "" || res.Digest == "" {
		t.Fatalf("unexpected result %+v", res)
	}
}

func TestEditingAnyValueIsDetected(t *testing.T) {
	edits := map[string]func(r *report.Report){
		"hide an uncorrected ECC error": func(r *report.Report) { r.GPUs[0].Memory.Aggregate.Uncorrected = report.Ptr(uint64(1)) },
		"lower a temperature":           func(r *report.Report) { r.GPUs[0].Samples[0].TempC = report.Ptr(int64(59)) },
		"swap the serial":               func(r *report.Report) { r.GPUs[0].Identity.Serial = "S2" },
		"delete an event":               func(r *report.Report) { r.GPUs[0].Events = nil },
		"backdate the report":           func(r *report.Report) { r.CreatedAt = "2025-01-01T00:00:00Z" },
	}
	for name, edit := range edits {
		r := sampleReport()
		_ = Sign(&r, testKey(1))
		edit(&r)
		if _, err := Verify(r, nil); !errors.Is(err, ErrTampered) {
			t.Errorf("%s: got %v, want ErrTampered", name, err)
		}
	}
}

func TestReSigningWithAnotherKeyFailsPinnedCheck(t *testing.T) {
	// A forger edits the report and re-signs it with their own key. The signature
	// itself is valid, but it doesn't match the key we trust.
	r := sampleReport()
	_ = Sign(&r, testKey(1))
	trusted := testKey(1).Public().(ed25519.PublicKey)

	r.GPUs[0].Memory.Aggregate.Uncorrected = report.Ptr(uint64(0))
	_ = Sign(&r, testKey(2))

	if _, err := Verify(r, nil); err != nil {
		t.Fatalf("unpinned verify should pass (that's why pinning matters): %v", err)
	}
	if _, err := Verify(r, trusted); err == nil {
		t.Fatal("pinned verify should reject a report signed by a different key")
	}
}

func TestUnsignedReportFails(t *testing.T) {
	if _, err := Verify(sampleReport(), nil); err == nil {
		t.Fatal("want error for unsigned report")
	}
}

func TestSignaturesAreDeterministic(t *testing.T) {
	a, b := sampleReport(), sampleReport()
	_ = Sign(&a, testKey(1))
	_ = Sign(&b, testKey(1))
	if *a.Signature != *b.Signature {
		t.Fatal("Ed25519 signatures of identical reports should be identical")
	}
}

func TestKeyFilesRoundTrip(t *testing.T) {
	dir := t.TempDir()
	priv := testKey(3)
	if err := SaveKeyPair(dir, priv, false); err != nil {
		t.Fatal(err)
	}
	gotPriv, err := LoadPrivateKey(filepath.Join(dir, PrivateKeyFile))
	if err != nil || !gotPriv.Equal(priv) {
		t.Fatalf("private key round trip failed: %v", err)
	}
	gotPub, err := LoadPublicKey(filepath.Join(dir, PublicKeyFile))
	if err != nil || !gotPub.Equal(priv.Public()) {
		t.Fatalf("public key round trip failed: %v", err)
	}
	info, _ := os.Stat(filepath.Join(dir, PrivateKeyFile))
	if info.Mode().Perm() != 0o600 {
		t.Fatalf("private key mode %v, want 0600", info.Mode().Perm())
	}
}

func TestSaveKeyPairWontOverwrite(t *testing.T) {
	dir := t.TempDir()
	_ = SaveKeyPair(dir, testKey(1), false)
	if err := SaveKeyPair(dir, testKey(2), false); err == nil {
		t.Fatal("want error when a key already exists")
	}
	if err := SaveKeyPair(dir, testKey(2), true); err != nil {
		t.Fatalf("force should overwrite: %v", err)
	}
}

// A real report from a Colab Tesla T4, signed on 8 Nov 2026 by the Week 1 agent,
// before probes, host_probes and per-reading PCIe fields existed. It must keep
// verifying no matter how the report format grows.
func TestRealWeek1ReportStillVerifies(t *testing.T) {
	data, err := os.ReadFile("testdata/colab-t4-2026-11-08.json")
	if err != nil {
		t.Fatal(err)
	}
	res, sig, err := VerifyJSON(data, nil)
	if err != nil {
		t.Fatalf("real Week 1 report no longer verifies: %v", err)
	}
	if res.KeyID != "6d1a4208fab0fc24" || sig.Digest != res.Digest {
		t.Fatalf("result %+v", res)
	}
	// One changed digit must still be caught.
	edited := bytes.Replace(data, []byte(`"temp_c": 52`), []byte(`"temp_c": 51`), 1)
	if _, _, err := VerifyJSON(edited, nil); !errors.Is(err, ErrTampered) {
		t.Fatalf("edited real report: got %v, want ErrTampered", err)
	}
}

func TestUnknownFieldsAreCoveredBySignature(t *testing.T) {
	// A field added by a future version is signed too: removing it must fail.
	r := sampleReport()
	_ = Sign(&r, testKey(1))
	data, _ := json.Marshal(r)
	var doc map[string]any
	_ = json.Unmarshal(data, &doc)
	doc["future_field"] = "x"
	delete(doc, "signature")
	msg, _ := report.Canonical(doc)
	sum := sha256.Sum256(msg)
	doc["signature"] = map[string]any{"algorithm": "ed25519", "key_id": KeyID(testKey(1).Public().(ed25519.PublicKey)),
		"public_key": base64.StdEncoding.EncodeToString(testKey(1).Public().(ed25519.PublicKey)),
		"digest":     "sha256:" + hex.EncodeToString(sum[:]),
		"value":      base64.StdEncoding.EncodeToString(ed25519.Sign(testKey(1), msg))}
	withField, _ := json.Marshal(doc)
	if _, _, err := VerifyJSON(withField, nil); err != nil {
		t.Fatalf("report with a newer field should verify: %v", err)
	}
	delete(doc, "future_field")
	without, _ := json.Marshal(doc)
	if _, _, err := VerifyJSON(without, nil); !errors.Is(err, ErrTampered) {
		t.Fatal("dropping a signed field must be caught")
	}
}
