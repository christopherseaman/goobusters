(() => {
  const logEl = document.getElementById("log");
  const statusEl = document.getElementById("status");
  const seriesListEl = document.getElementById("series-list");
  const frameImg = document.getElementById("frame-img");
  const maskImg = document.getElementById("mask-img");
  const metaEl = document.getElementById("meta");

  let selected = null;

  const log = (msg) => {
    const ts = new Date().toISOString();
    logEl.textContent = `${ts} ${msg}\n${logEl.textContent}`;
  };

  const jsonFetch = async (url, opts = {}) => {
    const resp = await fetch(url, opts);
    const text = await resp.text();
    let body = {};
    try {
      body = text ? JSON.parse(text) : {};
    } catch {
      body = { raw: text };
    }
    return { status: resp.status, headers: resp.headers, body };
  };

  const setStatus = (msg) => {
    statusEl.textContent = msg;
  };

  const renderSeries = (series) => {
    seriesListEl.innerHTML = "";
    if (!series || !series.length) {
      seriesListEl.innerHTML = "<p>No series found</p>";
      return;
    }
    series.forEach((item) => {
      const div = document.createElement("div");
      div.className = "series-item";
      div.innerHTML = `
        <div><strong>${item.study_uid}</strong> / ${item.series_uid}</div>
        <div>Exam: ${item.exam_number ?? "-"} | Series #: ${item.series_number ?? "-"}</div>
        <button data-action="select">Select</button>
      `;
      div.querySelector('[data-action="select"]').onclick = () => selectSeries(item);
      seriesListEl.appendChild(div);
    });
  };

  const selectSeries = (item) => {
    selected = item;
    setStatus(`Selected ${item.study_uid} / ${item.series_uid}`);
    // Fetch masks and first frame preview
    previewFrame(item);
    loadMask(item);
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

  const previewFrame = async (item) => {
    const method = item.method || "dis";
    const url = `/proxy/api/frames/${method}/${item.study_uid}/${item.series_uid}`;
    log(`Fetching frames archive for ${item.study_uid}/${item.series_uid}`);
    const metaResp = await fetch(url);
    if (!metaResp.ok) {
      log(`Frames meta status ${metaResp.status}`);
      setStatus("Frames unavailable");
      return;
    }
    const { frames_archive_url } = await metaResp.json();
    if (!frames_archive_url) {
      log("No frames archive url returned");
      setStatus("Frames unavailable");
      return;
    }

    const framesUrl = frames_archive_url.startsWith("/api/")
      ? `/proxy${frames_archive_url}`
      : frames_archive_url;
    const archiveResp = await fetch(framesUrl);
    if (!archiveResp.ok) {
      log(`Frames archive status ${archiveResp.status}`);
      setStatus("Frames unavailable");
      return;
    }
    const buffer = await archiveResp.arrayBuffer();
    const previewUrl = parseFirstFrameUrl(buffer);
    if (!previewUrl) {
      log("Could not parse first frame from archive");
      setStatus("Frames unavailable");
      return;
    }

    frameImg.src = previewUrl;
    frameImg.onload = () => {
      frameImg.style.display = "block";
      setStatus(`Previewing ${item.study_uid}/${item.series_uid}`);
    };
  };

  const loadMask = async (item) => {
    const url = `/proxy/api/masks/${item.study_uid}/${item.series_uid}`;
    log(`Fetching masks via proxy for ${item.study_uid}/${item.series_uid}`);
    const resp = await fetch(url);
    if (resp.status !== 200) {
      const txt = await resp.text();
      log(`Masks fetch status ${resp.status}: ${txt}`);
      setStatus(`Masks unavailable (${resp.status})`);
      maskImg.style.display = "none";
      return;
    }
    const version = resp.headers.get("x-version-id") || "";
    const count = resp.headers.get("x-mask-count") || "";
    setStatus(`Masks ready (count=${count}, version=${version})`);
    // For quick overlay demo: render the first mask if present by unpacking the tar? (Not trivial in browser).
    // Instead, log success and rely on the real viewer.js to load masks; here we just confirm fetch.
    log(`Masks downloaded (${count} masks, version=${version})`);
  };

  document.getElementById("btn-sync").onclick = async () => {
    setStatus("Syncing dataset...");
    const { status, body } = await jsonFetch("/api/dataset/sync", { method: "POST" });
    log(`Sync status ${status}: ${JSON.stringify(body)}`);
    if (status === 200 && body.series_count !== undefined) {
      setStatus(`Sync complete: ${body.series_count} series`);
    } else if (status === 200) {
      setStatus("Sync complete");
    } else {
      setStatus(`Sync failed: ${body.error || "Unknown error"}`);
    }
  };

  document.getElementById("btn-load-series").onclick = async () => {
    const { status, body } = await jsonFetch("/api/local/series");
    log(`Series status ${status}, count=${body.length || 0}`);
    if (status === 200) renderSeries(body);
  };

  document.getElementById("btn-auto-frames").onclick = async () => {
    if (!selected) {
      setStatus("Select a series first");
      return;
    }
    await previewFrame(selected);
    setStatus("Frames ready");
  };
})();
