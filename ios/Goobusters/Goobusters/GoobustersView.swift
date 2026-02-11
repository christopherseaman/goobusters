import SwiftUI
import WebKit
import os.log

#if os(macOS)
import AppKit
#else
import UIKit
#endif

private let logger = Logger(subsystem: "com.goobusters.app", category: "WebView")

struct GoobustersView: View {
    @StateObject private var backendManager = BackendManager()
    
    var body: some View {
        ZStack {
            // Black background - always show to prevent white flash
            Color.black
                .ignoresSafeArea()
            
            // Always show WebView - frontend handles connection state
            // This prevents losing state when backend restarts
            WebView(url: backendManager.serverURL, isReady: backendManager.isReady)
                .ignoresSafeArea()
                .id("webview") // Use stable ID to prevent unnecessary reloads
                .background(Color.black) // Ensure black background
                .opacity(backendManager.isReady ? 1.0 : 0.0) // Hide when not ready, but keep alive
                .allowsHitTesting(backendManager.isReady) // Disable interaction when not ready
            
            // Show connection status overlay when backend is not ready
            if !backendManager.isReady {
                ZStack {
                    Color.black.opacity(0.8)
                        .ignoresSafeArea()
                    VStack(spacing: 16) {
                        Text("🔌")
                            .font(.system(size: 60))
                        Text(backendManager.errorMessage ?? backendManager.statusMessage)
                            .font(.headline)
                            .foregroundColor(.white)
                        if let error = backendManager.errorMessage {
                            Text(error)
                                .font(.caption)
                                .foregroundColor(.gray)
                                .multilineTextAlignment(.center)
                                .padding(.horizontal)
                        }
                    }
                }
            }
        }
        .preferredColorScheme(.dark)
        .background(Color.black) // Ensure black background
        .onAppear {
            backendManager.start()
        }
        .onDisappear {
            backendManager.stop()
        }
    }
}

#if os(macOS)
struct WebView: NSViewRepresentable {
    let url: URL
    let isReady: Bool

    func makeNSView(context: Context) -> WKWebView {
        let config = WKWebViewConfiguration()
        config.allowsInlineMediaPlayback = true
        config.mediaTypesRequiringUserActionForPlayback = []

        // Add JavaScript console logging
        let contentController = config.userContentController
        contentController.add(context.coordinator, name: "consoleLog")
        let consoleScript = WKUserScript(
            source: """
                (function() {
                    const originalLog = console.log;
                    const originalError = console.error;
                    const originalWarn = console.warn;
                    console.log = function(...args) {
                        window.webkit.messageHandlers.consoleLog.postMessage({level: 'log', message: args.map(String).join(' ')});
                        originalLog.apply(console, args);
                    };
                    console.error = function(...args) {
                        window.webkit.messageHandlers.consoleLog.postMessage({level: 'error', message: args.map(String).join(' ')});
                        originalError.apply(console, args);
                    };
                    console.warn = function(...args) {
                        window.webkit.messageHandlers.consoleLog.postMessage({level: 'warn', message: args.map(String).join(' ')});
                        originalWarn.apply(console, args);
                    };
                    window.onerror = function(msg, url, line, col, error) {
                        window.webkit.messageHandlers.consoleLog.postMessage({level: 'error', message: 'JS Error: ' + msg + ' at ' + url + ':' + line});
                        return false;
                    };
                })();
                """,
            injectionTime: .atDocumentStart,
            forMainFrameOnly: true
        )
        contentController.addUserScript(consoleScript)

        let webView = WKWebView(frame: .zero, configuration: config)
        webView.navigationDelegate = context.coordinator
        context.coordinator.webView = webView
        // Set black background to prevent white flash
        webView.setValue(false, forKey: "drawsBackground")
        // Don't load URL here - wait for updateNSView when isReady
        return webView
    }

    func updateNSView(_ webView: WKWebView, context: Context) {
        // Reload when backend becomes ready OR if URL changes
        let currentURL = webView.url?.absoluteString ?? ""
        let targetURL = url.absoluteString
        let needsLoad = currentURL != targetURL || (isReady && currentURL.isEmpty)

        if isReady && needsLoad {
            logger.info("WebView loading URL: \(targetURL) (isReady=\(self.isReady), currentURL=\(currentURL))")
            let request = URLRequest(url: url)
            webView.load(request)
        }
    }

    func makeCoordinator() -> Coordinator {
        Coordinator()
    }

    class Coordinator: NSObject, WKNavigationDelegate, WKScriptMessageHandler {
        weak var webView: WKWebView?

        func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!, withError error: Error) {
            logger.error("WebView navigation error: \(error.localizedDescription)")
        }

        func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
            logger.info("WebView finished loading: \(webView.url?.absoluteString ?? "unknown")")
        }

        func webViewWebContentProcessDidTerminate(_ webView: WKWebView) {
            logger.error("WebView content process terminated, reloading")
            webView.reload()
        }

        func userContentController(_ userContentController: WKUserContentController, didReceive message: WKScriptMessage) {
            if message.name == "consoleLog", let body = message.body as? [String: Any] {
                let level = body["level"] as? String ?? "log"
                let msg = body["message"] as? String ?? ""
                logger.info("[JS \(level)] \(msg)")
            }
        }
    }
}
#else
struct WebView: UIViewRepresentable {
    let url: URL
    let isReady: Bool

    func makeUIView(context: Context) -> WKWebView {
        let config = WKWebViewConfiguration()
        config.allowsInlineMediaPlayback = true
        config.mediaTypesRequiringUserActionForPlayback = []

        // Add JavaScript console logging
        let contentController = config.userContentController
        contentController.add(context.coordinator, name: "consoleLog")
        let consoleScript = WKUserScript(
            source: """
                (function() {
                    const originalLog = console.log;
                    const originalError = console.error;
                    const originalWarn = console.warn;
                    console.log = function(...args) {
                        window.webkit.messageHandlers.consoleLog.postMessage({level: 'log', message: args.map(String).join(' ')});
                        originalLog.apply(console, args);
                    };
                    console.error = function(...args) {
                        window.webkit.messageHandlers.consoleLog.postMessage({level: 'error', message: args.map(String).join(' ')});
                        originalError.apply(console, args);
                    };
                    console.warn = function(...args) {
                        window.webkit.messageHandlers.consoleLog.postMessage({level: 'warn', message: args.map(String).join(' ')});
                        originalWarn.apply(console, args);
                    };
                    window.onerror = function(msg, url, line, col, error) {
                        window.webkit.messageHandlers.consoleLog.postMessage({level: 'error', message: 'JS Error: ' + msg + ' at ' + url + ':' + line});
                        return false;
                    };
                })();
                """,
            injectionTime: .atDocumentStart,
            forMainFrameOnly: true
        )
        contentController.addUserScript(consoleScript)

        let webView = WKWebView(frame: .zero, configuration: config)
        webView.navigationDelegate = context.coordinator
        context.coordinator.webView = webView
        // Set black background to prevent white flash
        webView.backgroundColor = .black
        webView.isOpaque = false
        webView.scrollView.backgroundColor = .black
        // Prevent WKWebView from adding its own content insets for safe areas
        webView.scrollView.contentInsetAdjustmentBehavior = .never
        // Don't load URL here - wait for updateUIView when isReady
        return webView
    }

    func updateUIView(_ webView: WKWebView, context: Context) {
        // Reload when backend becomes ready OR if URL changes
        let currentURL = webView.url?.absoluteString ?? ""
        let targetURL = url.absoluteString
        let needsLoad = currentURL != targetURL || (isReady && currentURL.isEmpty)

        if isReady && needsLoad {
            logger.info("WebView loading URL: \(targetURL) (isReady=\(self.isReady), currentURL=\(currentURL))")
            let request = URLRequest(url: url)
            webView.load(request)
        }
    }

    func makeCoordinator() -> Coordinator {
        Coordinator()
    }

    class Coordinator: NSObject, WKNavigationDelegate, WKScriptMessageHandler {
        weak var webView: WKWebView?

        override init() {
            super.init()
            NotificationCenter.default.addObserver(
                self,
                selector: #selector(appDidBecomeActive),
                name: UIApplication.didBecomeActiveNotification,
                object: nil
            )
        }

        deinit {
            NotificationCenter.default.removeObserver(self)
        }

        @objc func appDidBecomeActive() {
            logger.info("App became active, notifying WebView")
            webView?.evaluateJavaScript("if (window.viewer) viewer.handleAppForeground();") { _, error in
                if let error = error {
                    logger.error("Foreground JS call failed: \(error.localizedDescription)")
                }
            }
        }

        func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!, withError error: Error) {
            logger.error("WebView navigation error: \(error.localizedDescription)")
        }

        func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
            logger.info("WebView finished loading: \(webView.url?.absoluteString ?? "unknown")")
        }

        // Recover from iOS killing the WKWebView content process (memory pressure)
        func webViewWebContentProcessDidTerminate(_ webView: WKWebView) {
            logger.error("WebView content process terminated, reloading")
            webView.reload()
        }

        func userContentController(_ userContentController: WKUserContentController, didReceive message: WKScriptMessage) {
            if message.name == "consoleLog", let body = message.body as? [String: Any] {
                let level = body["level"] as? String ?? "log"
                let msg = body["message"] as? String ?? ""
                logger.info("[JS \(level)] \(msg)")
            }
        }
    }
}
#endif