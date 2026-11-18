// Hover crosshair + tooltip for the line charts. The chart is drawn server-side;
// this only snaps a hairline to the nearest reading and shows its value.
// Values are written with textContent (report data is never parsed as HTML).
document.querySelectorAll("figure.chart[data-series]").forEach((fig) => {
  const data = JSON.parse(fig.dataset.series);
  const svg = fig.querySelector("svg");
  const cross = svg.querySelector(".cross");
  const tip = fig.querySelector(".tip");
  const strong = document.createElement("strong");
  const when = document.createElement("span");
  tip.append(strong, when);

  const show = (evt) => {
    const box = svg.getBoundingClientRect();
    const vb = svg.viewBox.baseVal;
    const px = ((evt.clientX - box.left) / box.width) * vb.width;
    let i = 0;
    for (let k = 1; k < data.x.length; k++) if (Math.abs(data.x[k] - px) < Math.abs(data.x[i] - px)) i = k;
    cross.setAttribute("x1", data.x[i]);
    cross.setAttribute("x2", data.x[i]);
    cross.setAttribute("visibility", "visible");
    strong.textContent = `${data.v[i].toLocaleString()} ${data.unit}`;
    when.textContent = `at ${(data.t[i] / 1000).toFixed(1)} s`;
    tip.hidden = false;
    const left = (data.x[i] / vb.width) * box.width;
    tip.style.left = `${Math.min(Math.max(left + 12, 0), box.width - tip.offsetWidth)}px`;
  };
  const hide = () => { cross.setAttribute("visibility", "hidden"); tip.hidden = true; };
  svg.addEventListener("pointermove", show);
  svg.addEventListener("pointerleave", hide);
});
