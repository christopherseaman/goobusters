import Foundation

@MainActor
final class PythonBackendRunner {
    enum RunnerError: LocalizedError {
        case bundleMissing
        case entryScriptMissing(String)
        case pythonInitializationFailed(String)

        var errorDescription: String? {
            switch self {
            case .bundleMissing:
                return "Unable to resolve bundle resource path"
            case .entryScriptMissing(let path):
                return "Backend entry script missing: \(path)"
            case .pythonInitializationFailed(let details):
                return "Python initialization failed: \(details)"
            }
        }
    }

    private var runner: BackendPythonRunner?
    private let resourcePath: String

    init(resourcePath: String) {
        self.resourcePath = resourcePath
    }

    func start(entryScriptRelativePath: String) async throws {
        // If runner exists and is running, skip
        if runner?.isRunning() == true { return }

        let entryFullPath = (resourcePath as NSString).appendingPathComponent(entryScriptRelativePath)
        guard FileManager.default.fileExists(atPath: entryFullPath) else {
            throw RunnerError.entryScriptMissing(entryFullPath)
        }

        let backendRunner = BackendPythonRunner(resourcePath: resourcePath)
        do {
            try backendRunner.start(withEntryScript: entryFullPath)
        } catch {
            throw RunnerError.pythonInitializationFailed(error.localizedDescription)
        }

        runner = backendRunner
    }

    func stop() async {
        runner?.stop()
        runner = nil
    }
}
