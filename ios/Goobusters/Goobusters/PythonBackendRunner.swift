import Foundation

#if os(macOS)
import AppKit
#endif

@MainActor
final class PythonBackendRunner {
    enum RunnerError: LocalizedError {
        case bundleMissing
        case entryScriptMissing(String)
        case pythonInitializationFailed(String)
        case systemPythonNotFound

        var errorDescription: String? {
            switch self {
            case .bundleMissing:
                return "Unable to resolve bundle resource path"
            case .entryScriptMissing(let path):
                return "Backend entry script missing: \(path)"
            case .pythonInitializationFailed(let details):
                return "Python initialization failed: \(details)"
            case .systemPythonNotFound:
                return "System Python not found. Please install Python 3."
            }
        }
    }

    private var runner: BackendPythonRunner?
    #if os(macOS) || targetEnvironment(macCatalyst)
    private var process: Process?
    #endif
    private let resourcePath: String

    init(resourcePath: String) {
        self.resourcePath = resourcePath
    }

    func start(entryScriptRelativePath: String) async throws {
        #if os(macOS) || targetEnvironment(macCatalyst)
        // On macOS or Mac Catalyst, use system Python via Process
        try await startWithSystemPython(entryScriptRelativePath: entryScriptRelativePath)
        #else
        // On iOS, use embedded Python framework
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
        #endif
    }

    #if os(macOS) || targetEnvironment(macCatalyst)
    private func startWithSystemPython(entryScriptRelativePath: String) async throws {
        // Find system Python
        let pythonPaths = [
            "/usr/bin/python3",
            "/usr/local/bin/python3",
            "/opt/homebrew/bin/python3",
            "/Library/Frameworks/Python.framework/Versions/Current/bin/python3"
        ]
        
        var pythonPath: String?
        for path in pythonPaths {
            if FileManager.default.fileExists(atPath: path) {
                pythonPath = path
                break
            }
        }
        
        // Try to find python3 in PATH
        if pythonPath == nil {
            let task = Process()
            task.launchPath = "/usr/bin/which"
            task.arguments = ["python3"]
            let pipe = Pipe()
            task.standardOutput = pipe
            task.standardError = Pipe()
            
            do {
                try task.run()
                task.waitUntilExit()
                
                if task.terminationStatus == 0 {
                    let data = pipe.fileHandleForReading.readDataToEndOfFile()
                    if let output = String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines),
                       !output.isEmpty {
                        pythonPath = output
                    }
                }
            } catch {
                // Ignore errors
            }
        }
        
        guard let pythonPath = pythonPath else {
            throw RunnerError.systemPythonNotFound
        }
        
        let entryFullPath = (resourcePath as NSString).appendingPathComponent(entryScriptRelativePath)
        guard FileManager.default.fileExists(atPath: entryFullPath) else {
            throw RunnerError.entryScriptMissing(entryFullPath)
        }
        
        // Set up environment
        var environment = ProcessInfo.processInfo.environment
        environment["PYTHONPATH"] = resourcePath
        environment["DATA_DIR"] = NSSearchPathForDirectoriesInDomains(.applicationSupportDirectory, .userDomainMask, true).first ?? resourcePath
        
        // Create process
        let process = Process()
        process.executableURL = URL(fileURLWithPath: pythonPath)
        process.arguments = [entryFullPath]
        process.environment = environment
        process.currentDirectoryURL = URL(fileURLWithPath: resourcePath)
        
        // Redirect output
        let outputPipe = Pipe()
        let errorPipe = Pipe()
        process.standardOutput = outputPipe
        process.standardError = errorPipe
        
        // Handle output asynchronously
        outputPipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            if !data.isEmpty {
                if let output = String(data: data, encoding: .utf8) {
                    print("[Python] \(output)")
                }
            }
        }
        
        errorPipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            if !data.isEmpty {
                if let output = String(data: data, encoding: .utf8) {
                    print("[Python Error] \(output)")
                }
            }
        }
        
        do {
            try process.run()
            self.process = process
        } catch {
            throw RunnerError.pythonInitializationFailed(error.localizedDescription)
        }
    }
    #endif

    func stop() async {
        #if os(macOS) || targetEnvironment(macCatalyst)
        process?.terminate()
        process?.waitUntilExit()
        process = nil
        #else
        runner?.stop()
        runner = nil
        #endif
    }
    
    func isRunning() -> Bool {
        #if os(macOS) || targetEnvironment(macCatalyst)
        return process?.isRunning ?? false
        #else
        return runner?.isRunning() ?? false
        #endif
    }
}
