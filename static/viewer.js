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
    // Auto ensure frames
    ensureFrames(item);
    // Fetch masks and first frame
    loadFrame(item, 0);
    loadMask(item);
  };

  const ensureFrames = async (item) => {
    const url = `/api/local/frames/${item.study_uid}/${item.series_uid}`;
    log(`Extracting frames for ${item.study_uid}/${item.series_uid}`);
    await jsonFetch(url, { method: "POST" });
  };

  const loadFrame = (item, index) => {
    frameImg.src = `/frames/${item.study_uid}/${item.series_uid}/${index}.png`;
    frameImg.onload = () => {
      frameImg.style.display = "block";
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
    setStatus(`Sync complete (status ${status})`);
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
    await ensureFrames(selected);
    setStatus("Frames ready");
  };
})();

