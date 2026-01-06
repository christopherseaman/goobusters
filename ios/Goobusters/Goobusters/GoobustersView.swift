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
                    WebView(url: URL(string: "http://localhost:8080")!)
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
        return webView
    }
    
    func updateUIView(_ webView: WKWebView, context: Context) {
        if webView.url != url {
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
    }
}

