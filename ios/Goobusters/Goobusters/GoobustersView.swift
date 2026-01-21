import SwiftUI
import WebKit

struct GoobustersView: View {
    @StateObject private var backendManager = BackendManager()
    
    var body: some View {
        ZStack {
            // Black background for top/bottom bars
            Color.black
                .ignoresSafeArea()
            
            Group {
                if backendManager.isReady {
                    WebView(url: backendManager.serverURL)
                        .id(backendManager.serverURL.absoluteString) // Force recreation when URL changes
                } else if let error = backendManager.errorMessage {
                    VStack(spacing: 16) {
                        Image(systemName: "exclamationmark.triangle")
                            .font(.largeTitle)
                            .foregroundColor(.red)
                        Text("Backend Error")
                            .font(.headline)
                            .foregroundColor(.white)
                        Text(error)
                            .font(.caption)
                            .foregroundColor(.gray)
                            .multilineTextAlignment(.center)
                            .padding(.horizontal)
                    }
                } else {
                    VStack(spacing: 16) {
                        ProgressView()
                            .tint(.white)
                        Text(backendManager.statusMessage)
                            .font(.caption)
                            .foregroundColor(.white)
                    }
                }
            }
        }
        .preferredColorScheme(.dark)
        .onAppear {
            backendManager.start()
        }
        .onDisappear {
            backendManager.stop()
        }
    }
}

struct WebView: UIViewRepresentable {
    let url: URL
    
    func makeUIView(context: Context) -> WKWebView {
        let config = WKWebViewConfiguration()
        config.allowsInlineMediaPlayback = true
        config.mediaTypesRequiringUserActionForPlayback = []
        
        let webView = WKWebView(frame: .zero, configuration: config)
        webView.navigationDelegate = context.coordinator
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