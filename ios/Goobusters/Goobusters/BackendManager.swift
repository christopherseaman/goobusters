import Foundation
import Combine

@MainActor
class BackendManager: ObservableObject {
    @Published var isReady = false
    @Published var statusMessage = "Starting backend..."
    
    private var checkTimer: Timer?
    
    func start() {
        statusMessage = "Starting Python backend..."
        
        // TODO: Launch Python backend
        // For now, just mark as ready after a delay
        // In production, this will:
        // 1. Check if Python runtime is available (Pyto or Pyodide)
        // 2. Launch lib/client/start.py
        // 3. Wait for http://localhost:8080 to be ready
        // 4. Set isReady = true
        
        DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
            self.statusMessage = "Backend ready"
            self.isReady = true
        }
    }
    
    func stop() {
        checkTimer?.invalidate()
        checkTimer = nil
    }
    
    private func checkBackendReady() {
        guard let url = URL(string: "http://localhost:8080/healthz") else { return }
        
        var request = URLRequest(url: url)
        request.timeoutInterval = 1.0
        
        URLSession.shared.dataTask(with: request) { [weak self] data, response, error in
            DispatchQueue.main.async {
                if let httpResponse = response as? HTTPURLResponse,
                   httpResponse.statusCode == 200 {
                    self?.isReady = true
                    self?.statusMessage = "Backend ready"
                    self?.checkTimer?.invalidate()
                } else {
                    self?.statusMessage = "Waiting for backend..."
                }
            }
        }.resume()
    }
}

