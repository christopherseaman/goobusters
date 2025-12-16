(() => {
  const logEl = document.getElementById("log");
  const healthEl = document.getElementById("health-status");
  const seriesListEl = document.getElementById("series-list");
  const previewEl = document.getElementById("preview");

  const log = (msg) => {
    const ts = new Date().toISOString();
    logEl.textContent = `${ts} ${msg}\n${logEl.textContent}`;
  };

  const jsonFetch = async (url, opts = {}) => {
    const resp = await fetch(url, opts);
    const text = await resp.text();
    let body;
    try {
      body = text ? JSON.parse(text) : {};
    } catch (_err) {
      body = { raw: text };
    }
    return { status: resp.status, headers: resp.headers, body };
  };

  const renderSeries = (series) => {
    if (!series || !series.length) {
      seriesListEl.innerHTML = "<p>No series found</p>";
      return;
    }
    seriesListEl.innerHTML = "";
    series.forEach((item) => {
      const div = document.createElement("div");
      div.style.border = "1px solid #ccc";
      div.style.padding = "8px";
      div.style.margin = "8px 0";
      div.innerHTML = `
        <div><strong>${item.study_uid}</strong> / ${item.series_uid}</div>
        <div>Exam: ${item.exam_number ?? "-"} | Series #: ${item.series_number ?? "-"}</div>
        <div>Video: ${item.video_path}</div>
        <button data-action="frames">Extract Frames</button>
        <button data-action="mask">Download Masks (server)</button>
      `;
      div.querySelector('[data-action="frames"]').onclick = () => extractFrames(item);
      div.querySelector('[data-action="mask"]').onclick = () => downloadMasks(item);
      seriesListEl.appendChild(div);
    });
  };

  const extractFrames = async (item) => {
    const url = `/api/local/frames/${item.study_uid}/${item.series_uid}`;
    log(`Extracting frames for ${item.study_uid}/${item.series_uid}`);
    const { status, body } = await jsonFetch(url, { method: "POST" });
    log(`Frames status ${status}`);
    if (status !== 200) return;
    const img = document.createElement("img");
    img.src = `/frames/${item.study_uid}/${item.series_uid}/0.png`;
    img.alt = "Frame 0";
    img.style.maxWidth = "480px";
    previewEl.innerHTML = "";
    previewEl.appendChild(img);
  };

  const downloadMasks = async (item) => {
    const url = `/proxy/api/masks/${item.study_uid}/${item.series_uid}`;
    log(`Downloading masks via proxy for ${item.study_uid}/${item.series_uid}`);
    const resp = await fetch(url);
    log(
      `Masks status ${resp.status} headers: version=${resp.headers.get(
        "x-version-id"
      )} count=${resp.headers.get("x-mask-count")}`
    );
    if (resp.status === 200) {
      const blob = await resp.blob();
      const sizeKb = Math.round(blob.size / 1024);
      log(`Downloaded ${sizeKb} KB`);
    } else {
      const text = await resp.text();
      log(`Non-200 response: ${text}`);
    }
  };

  document.getElementById("btn-health").onclick = async () => {
    const { status, body } = await jsonFetch("/healthz");
    healthEl.textContent = `status ${status} ready=${body.client_ready} server=${body.server_url}`;
    log(`Health: ${JSON.stringify(body)}`);
  };

  document.getElementById("btn-sync").onclick = async () => {
    log("Syncing dataset...");
    const { status, body } = await jsonFetch("/api/dataset/sync", {
      method: "POST",
    });
    log(`Sync status ${status}: ${JSON.stringify(body)}`);
  };

  document.getElementById("btn-load-series").onclick = async () => {
    const { status, body } = await jsonFetch("/api/local/series");
    log(`Series status ${status}, count=${body.length || 0}`);
    if (status === 200) renderSeries(body);
  };
})();

