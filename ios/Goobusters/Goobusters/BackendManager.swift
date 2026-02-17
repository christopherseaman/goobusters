import Foundation
import Combine

@MainActor
class BackendManager: ObservableObject {
    @Published var isReady = false
    @Published var needsReload = false
    @Published var statusMessage = "Starting backend..."
    @Published var errorMessage: String?

    private var pythonRunner: PythonBackendRunner?
    private var healthCheckTask: Task<Void, Never>?
    private var isStopped = false
    private let port = 8080
    private let entryScript = "python-app/start_server.py"

    private var resourcePath: String {
        #if os(macOS) || targetEnvironment(macCatalyst)
        // On macOS or Mac Catalyst, use bundle path or fallback to executable path
        if let bundlePath = Bundle.main.resourcePath {
            return bundlePath
        } else if let executablePath = Bundle.main.executablePath {
            return (executablePath as NSString).deletingLastPathComponent
        } else {
            return ""
        }
        #else
        return Bundle.main.resourcePath ?? ""
        #endif
    }

    func start() {
        let path = resourcePath
        guard !path.isEmpty else {
            errorMessage = "Bundle resource path not found"
            statusMessage = "Error: Bundle missing"
            return
        }

        isStopped = false
        statusMessage = "Connecting..."

        pythonRunner = PythonBackendRunner(resourcePath: path)

        Task {
            do {
                try await pythonRunner?.start(entryScriptRelativePath: entryScript)
                statusMessage = "Connecting..."

                // Poll for server readiness
                await waitForServer()

                // Start continuous health monitoring
                startHealthMonitoring()
            } catch {
                errorMessage = error.localizedDescription
                statusMessage = "Error: \(error.localizedDescription)"
            }
        }
    }

    private func waitForServer() async {
        let maxAttempts = 30
        let url = URL(string: "http://127.0.0.1:\(port)/healthz")!

        for _ in 1...maxAttempts {
            do {
                let (_, response) = try await URLSession.shared.data(from: url)
                if let httpResponse = response as? HTTPURLResponse,
                   httpResponse.statusCode == 200 {
                    isReady = true
                    errorMessage = nil
                    return
                }
            } catch {
                // Server not ready yet
            }

            try? await Task.sleep(nanoseconds: 500_000_000) // 0.5s
        }

        errorMessage = "Backend Error"
        statusMessage = "Backend Error"
    }

    private func startHealthMonitoring() {
        healthCheckTask?.cancel()

        healthCheckTask = Task {
            let url = URL(string: "http://127.0.0.1:\(port)/healthz")!
            var hadFailure = false

            while !isStopped && !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 2_000_000_000) // 2 seconds

                // Fresh session each time — stale connections fail after sleep
                let session = URLSession(configuration: {
                    let config = URLSessionConfiguration.ephemeral
                    config.timeoutIntervalForRequest = 5
                    config.timeoutIntervalForResource = 5
                    return config
                }())

                var ok = false
                do {
                    let (_, response) = try await session.data(from: url)
                    if let httpResponse = response as? HTTPURLResponse,
                       httpResponse.statusCode == 200 {
                        ok = true
                    }
                } catch {
                    // Network briefly unavailable (e.g. after sleep) — not a real failure
                }

                session.invalidateAndCancel()

                if ok && hadFailure {
                    // Backend recovered after a blip — tell JS to clear stale UI
                    hadFailure = false
                    needsReload = true
                } else if !ok {
                    hadFailure = true
                }
            }
        }
    }

    func stop() {
        isStopped = true
        healthCheckTask?.cancel()
        healthCheckTask = nil

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
