import Combine
import Darwin
import Foundation

@MainActor
final class BackendManager: ObservableObject {
    @Published var isReady = false
    @Published var statusMessage = "Starting backend..."

    private var pythonRunner: PythonBackendRunner?
    private var checkTimer: Timer?
    private let healthCheckURL = URL(string: "http://localhost:8080/healthz")
    private let backendEntryPoint = "lib/client/start.py"

    func start() {
        guard checkTimer == nil else { return }
        guard let bundlePath = Bundle.main.resourcePath else {
            statusMessage = "Unable to determine bundle path"
            return
        }

        statusMessage = "Starting Python backend..."

        configurePythonEnvironment(bundlePath: bundlePath)

        Task { [weak self] in
            guard let self else { return }

            do {
                if self.pythonRunner == nil {
                    self.pythonRunner = PythonBackendRunner(resourcePath: bundlePath)
                }

                try await self.pythonRunner?.start(entryScriptRelativePath: backendEntryPoint)
                self.startHealthChecks()
            } catch {
                self.statusMessage = "Python start failed: \(error.localizedDescription)"
            }
        }
    }

    private func configurePythonEnvironment(bundlePath: String) {
        let pythonHome = bundlePath + "/python"
        var searchPaths: [String] = []

        if let stdlibPath = pythonStdlibPath(pythonHome: pythonHome) {
            searchPaths.append(stdlibPath)
        }

        searchPaths.append(pythonHome + "/lib-dynload")
        searchPaths.append(bundlePath + "/lib/client")
        searchPaths.append(bundlePath)

        let pythonPath = searchPaths.joined(separator: ":")

        setEnvironmentVariable("PYTHONHOME", value: pythonHome)
        setEnvironmentVariable("PYTHONPATH", value: pythonPath)
        setEnvironmentVariable("PYTHONUNBUFFERED", value: "1")
        setEnvironmentVariable("PYTHONDONTWRITEBYTECODE", value: "1")
    }

    private func pythonStdlibPath(pythonHome: String) -> String? {
        let libDirectory = URL(fileURLWithPath: pythonHome).appendingPathComponent("lib", isDirectory: true)

        guard
            let contents = try? FileManager.default.contentsOfDirectory(atPath: libDirectory.path),
            let versionDirectory = contents
                .filter({ $0.hasPrefix("python3") })
                .sorted(by: >)
                .first
        else {
            return nil
        }

        return libDirectory.appendingPathComponent(versionDirectory, isDirectory: true).path
    }

    private func setEnvironmentVariable(_ key: String, value: String) {
        key.withCString { keyPointer in
            value.withCString { valuePointer in
                setenv(keyPointer, valuePointer, 1)
            }
        }
    }

    private func startHealthChecks() {
        checkTimer?.invalidate()

        checkTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] timer in
            guard let self else {
                timer.invalidate()
                return
            }

            Task { await self.pollBackendReady() }
        }

        if let checkTimer {
            RunLoop.main.add(checkTimer, forMode: .common)
        }
    }

    private func pollBackendReady() async {
        guard let healthCheckURL else { return }

        var request = URLRequest(url: healthCheckURL)
        request.timeoutInterval = 1.0

        do {
            let (_, response) = try await URLSession.shared.data(for: request)

            guard
                let httpResponse = response as? HTTPURLResponse,
                httpResponse.statusCode == 200
            else {
                statusMessage = "Waiting for backend..."
                return
            }

            isReady = true
            statusMessage = "Backend ready"
            checkTimer?.invalidate()
            checkTimer = nil
        } catch {
            statusMessage = "Waiting for backend..."
        }
    }

    func stop() {
        checkTimer?.invalidate()
        checkTimer = nil

        Task {
            await pythonRunner?.stop()
            pythonRunner = nil
        }
    }
}
