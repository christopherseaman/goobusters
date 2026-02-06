import Foundation
import Combine

@MainActor
class BackendManager: ObservableObject {
    @Published var isReady = false
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
            var consecutiveFailures = 0
            let maxFailures = 3

            let session = URLSession(configuration: {
                let config = URLSessionConfiguration.ephemeral
                config.timeoutIntervalForRequest = 3
                config.timeoutIntervalForResource = 3
                return config
            }())

            while !isStopped && !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 1_000_000_000) // 1 second

                let isRunning = await pythonRunner?.isRunning() ?? false
                if !isRunning {
                    consecutiveFailures = 0
                    await handleBackendCrash()
                    continue
                }

                do {
                    let (_, response) = try await session.data(from: url)
                    if let httpResponse = response as? HTTPURLResponse,
                       httpResponse.statusCode == 200 {
                        consecutiveFailures = 0
                        if !isReady {
                            isReady = true
                            errorMessage = nil
                        }
                    } else {
                        consecutiveFailures += 1
                    }
                } catch {
                    consecutiveFailures += 1
                }

                if consecutiveFailures >= maxFailures && isReady {
                    isReady = false
                    statusMessage = "Reconnecting..."
                }
            }
        }
    }
    
    private func handleBackendCrash() async {
        // Backend process died - restart it
        isReady = false
        statusMessage = "Restarting backend..."
        errorMessage = nil
        
        // Stop the old runner
        await pythonRunner?.stop()
        
        // Wait a moment before restarting
        try? await Task.sleep(nanoseconds: 1_000_000_000) // 1 second
        
        // Restart
        let path = resourcePath
        guard !path.isEmpty else {
            errorMessage = "Bundle resource path not found"
            statusMessage = "Error: Bundle missing"
            return
        }
        
        pythonRunner = PythonBackendRunner(resourcePath: path)
        
        do {
            try await pythonRunner?.start(entryScriptRelativePath: entryScript)
            statusMessage = "Connecting..."
            await waitForServer()
            // Health monitoring will continue automatically
        } catch {
            errorMessage = error.localizedDescription
            statusMessage = "Error: \(error.localizedDescription)"
            // Try again in a few seconds
            try? await Task.sleep(nanoseconds: 3_000_000_000)
            await handleBackendCrash()
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
