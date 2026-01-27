import SwiftUI
import WebKit

#if os(macOS)
import AppKit
#else
import UIKit
#endif

struct GoobustersView: View {
    @StateObject private var backendManager = BackendManager()
    
    var body: some View {
        ZStack {
            // Black background - always show to prevent white flash
            Color.black
                .ignoresSafeArea()
            
            // Always show WebView - frontend handles connection state
            // This prevents losing state when backend restarts
            WebView(url: backendManager.serverURL)
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
    
    func makeNSView(context: Context) -> WKWebView {
        let config = WKWebViewConfiguration()
        config.allowsInlineMediaPlayback = true
        config.mediaTypesRequiringUserActionForPlayback = []
        
        let webView = WKWebView(frame: .zero, configuration: config)
        webView.navigationDelegate = context.coordinator
        // Set black background to prevent white flash
        webView.setValue(false, forKey: "drawsBackground")
        // Load URL immediately when WebView is created
        let request = URLRequest(url: url)
        webView.load(request)
        return webView
    }
    
    func updateNSView(_ webView: WKWebView, context: Context) {
        // Always reload if URL is different or if webView has no URL yet
        let currentURL = webView.url?.absoluteString ?? ""
        let targetURL = url.absoluteString
        if currentURL != targetURL {
            let request = URLRequest(url: url)
            webView.load(request)
        }
    }
    
    func makeCoordinator() -> Coordinator {
        Coordinator()
    }
    
    class Coordinator: NSObject, WKNavigationDelegate {
        func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!, withError error: Error) {
            print("WebView navigation error: \(error.localizedDescription)")
        }
        
        func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
            print("WebView finished loading: \(webView.url?.absoluteString ?? "unknown")")
        }
    }
}
#else
struct WebView: UIViewRepresentable {
    let url: URL
    
    func makeUIView(context: Context) -> WKWebView {
        let config = WKWebViewConfiguration()
        config.allowsInlineMediaPlayback = true
        config.mediaTypesRequiringUserActionForPlayback = []
        
        let webView = WKWebView(frame: .zero, configuration: config)
        webView.navigationDelegate = context.coordinator
        // Set black background to prevent white flash
        webView.backgroundColor = .black
        webView.isOpaque = false
        webView.scrollView.backgroundColor = .black
        // Load URL immediately when WebView is created
        let request = URLRequest(url: url)
        webView.load(request)
        return webView
    }
    
    func updateUIView(_ webView: WKWebView, context: Context) {
        // Always reload if URL is different or if webView has no URL yet
        let currentURL = webView.url?.absoluteString ?? ""
        let targetURL = url.absoluteString
        if currentURL != targetURL {
            let request = URLRequest(url: url)
            webView.load(request)
        }
    }
    
    func makeCoordinator() -> Coordinator {
        Coordinator()
    }
    
    class Coordinator: NSObject, WKNavigationDelegate {
        func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!, withError error: Error) {
            print("WebView navigation error: \(error.localizedDescription)")
        }
        
        func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
            print("WebView finished loading: \(webView.url?.absoluteString ?? "unknown")")
        }
    }
}
#endif