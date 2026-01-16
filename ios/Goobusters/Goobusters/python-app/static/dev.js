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
        <div>Method: ${item.method ?? "dis"}</div>
        <button data-action="frames">Preview Frames</button>
        <button data-action="mask">Download Masks (server)</button>
      `;
      div.querySelector('[data-action="frames"]').onclick = () => previewFrames(item);
      div.querySelector('[data-action="mask"]').onclick = () => downloadMasks(item);
      seriesListEl.appendChild(div);
    });
  };

  const parseFirstFrameUrl = (arrayBuffer) => {
    const data = new Uint8Array(arrayBuffer);
    const decoder = new TextDecoder();
    let offset = 0;

    while (offset + 512 <= data.length) {
      const name = decoder
        .decode(data.slice(offset, offset + 100))
        .replace(/\0/g, "")
        .trim();
      if (!name) break;

      const sizeStr = decoder
        .decode(data.slice(offset + 124, offset + 136))
        .replace(/\0/g, "")
        .trim();
      const size = parseInt(sizeStr || "0", 8);
      offset += 512;

      if (size > 0 && /\.(webp|png)$/i.test(name)) {
        const fileBytes = data.slice(offset, offset + size);
        const mime = name.endsWith(".png") ? "image/png" : "image/webp";
        const blob = new Blob([fileBytes], { type: mime });
        return URL.createObjectURL(blob);
      }

      const padded = Math.ceil(size / 512) * 512;
      offset += padded;
    }
    return null;
  };

  const previewFrames = async (item) => {
    const method = item.method || "dis";
    const url = `/proxy/api/frames/${method}/${item.study_uid}/${item.series_uid}`;
    log(`Fetching frames archive for ${item.study_uid}/${item.series_uid}`);
    const metaResp = await fetch(url);
    if (!metaResp.ok) {
      log(`Frames meta status ${metaResp.status}`);
      return;
    }
    const { frames_archive_url } = await metaResp.json();
    if (!frames_archive_url) {
      log("No frames archive url returned");
      return;
    }

    const framesUrl = frames_archive_url.startsWith("/api/")
      ? `/proxy${frames_archive_url}`
      : frames_archive_url;
    const archiveResp = await fetch(framesUrl);
    if (!archiveResp.ok) {
      log(`Frames archive status ${archiveResp.status}`);
      return;
    }
    const buffer = await archiveResp.arrayBuffer();
    const previewUrl = parseFirstFrameUrl(buffer);
    if (!previewUrl) {
      log("Could not parse first frame from archive");
      return;
    }

    const img = document.createElement("img");
    img.src = previewUrl;
    img.alt = "Frame preview";
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
