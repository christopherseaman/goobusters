import Foundation
import Combine

@MainActor
class BackendManager: ObservableObject {
    @Published var isReady = false
    @Published var statusMessage = "Starting backend..."
    @Published var errorMessage: String?

    private var pythonRunner: PythonBackendRunner?
    private let port = 8080
    private let entryScript = "python-app/start_server.py"

    func start() {
        guard let resourcePath = Bundle.main.resourcePath else {
            errorMessage = "Bundle resource path not found"
            statusMessage = "Error: Bundle missing"
            return
        }

        statusMessage = "Initializing Python..."

        pythonRunner = PythonBackendRunner(resourcePath: resourcePath)

        Task {
            do {
                try await pythonRunner?.start(entryScriptRelativePath: entryScript)
                statusMessage = "Starting Flask server..."

                // Poll for server readiness
                await waitForServer()
            } catch {
                errorMessage = error.localizedDescription
                statusMessage = "Error: \(error.localizedDescription)"
            }
        }
    }

    private func waitForServer() async {
        let maxAttempts = 30
        let url = URL(string: "http://127.0.0.1:\(port)/healthz")!

        for attempt in 1...maxAttempts {
            do {
                let (_, response) = try await URLSession.shared.data(from: url)
                if let httpResponse = response as? HTTPURLResponse,
                   httpResponse.statusCode == 200 {
                    statusMessage = "Backend ready"
                    isReady = true
                    return
                }
            } catch {
                // Server not ready yet
            }

            try? await Task.sleep(nanoseconds: 500_000_000) // 0.5s
            statusMessage = "Waiting for server... (\(attempt)/\(maxAttempts))"
        }

        errorMessage = "Server failed to start after \(maxAttempts) attempts"
        statusMessage = "Error: Server timeout"
    }

    func stop() {
        Task {
            await pythonRunner?.stop()
        }
        isReady = false
        statusMessage = "Backend stopped"
    }

    var serverURL: URL {
        URL(string: "http://127.0.0.1:\(port)")!
    }
}
