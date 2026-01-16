#import "BackendPythonRunner.h"

#import <Python/Python.h>
#import <dispatch/dispatch.h>
#import <unistd.h>

static NSString *const BackendPythonRunnerErrorDomain = @"org.badmath.goobusters.python";

typedef NS_ENUM(NSInteger, BackendPythonRunnerErrorCode) {
    BackendPythonRunnerErrorCodeInitialization = 1,
};

static int BackendPythonRunnerRequestStop(void *arg) {
    PyErr_SetInterrupt();
    return 0;
}

@interface BackendPythonRunner ()
- (NSString *)pythonHomePath;
- (NSString *)programName;
- (NSError *)errorWithMessage:(NSString *)message;
- (NSError *)errorFromStatus:(PyStatus)status;
- (BOOL)configureInterpreterWithEntryScript:(NSString *)entryScript error:(NSError **)error;
- (void)runEntryScriptAtPath:(NSString *)entryScript;
- (void)updateWorkingDirectory;
@end

@implementation BackendPythonRunner {
    NSString *_resourcePath;
    BOOL _isRunning;
    dispatch_queue_t _pythonQueue;
}

- (instancetype)initWithResourcePath:(NSString *)resourcePath {
    self = [super init];
    if (self) {
        _resourcePath = [resourcePath copy];
        _isRunning = NO;
        _pythonQueue = dispatch_queue_create("org.badmath.goobusters.python", DISPATCH_QUEUE_SERIAL);
    }
    return self;
}

- (BOOL)startWithEntryScript:(NSString *)entryScript error:(NSError **)error {
    if (_isRunning) {
        return YES;
    }

    if (entryScript.length == 0) {
        if (error) {
            *error = [self errorWithMessage:@"Entry script path is empty"];
        }
        return NO;
    }

    dispatch_semaphore_t semaphore = dispatch_semaphore_create(0);
    __block NSError *startupError = nil;
    NSString *scriptPath = [entryScript copy];

    dispatch_async(_pythonQueue, ^{
        NSError *localError = nil;
        if (![self configureInterpreterWithEntryScript:scriptPath error:&localError]) {
            startupError = localError;
            dispatch_semaphore_signal(semaphore);
            return;
        }

        self->_isRunning = YES;
        dispatch_semaphore_signal(semaphore);
        [self runEntryScriptAtPath:scriptPath];
        self->_isRunning = NO;
    });

    long waitResult = dispatch_semaphore_wait(semaphore, dispatch_time(DISPATCH_TIME_NOW, (int64_t)(10 * NSEC_PER_SEC)));
    if (waitResult != 0) {
        if (error) {
            *error = [self errorWithMessage:@"Timed out starting Python backend"];
        }
        return NO;
    }

    if (startupError) {
        if (error) {
            *error = startupError;
        }
        return NO;
    }

    return YES;
}

- (void)stop {
    if (!_isRunning) {
        return;
    }

    if (!Py_IsInitialized()) {
        _isRunning = NO;
        return;
    }

    int result = Py_AddPendingCall(BackendPythonRunnerRequestStop, NULL);
    if (result != 0) {
        NSLog(@"[BackendPythonRunner] Unable to schedule Python interrupt");
    }
}

- (BOOL)isRunning {
    return _isRunning;
}

#pragma mark - Private helpers

- (BOOL)configureInterpreterWithEntryScript:(NSString *)entryScript error:(NSError **)error {
    PyConfig config;
    PyConfig_InitIsolatedConfig(&config);

    config.install_signal_handlers = 0;
    config.write_bytecode = 0;
    config.use_environment = 1;
    config.buffered_stdio = 0;
    config.parse_argv = 0;
    config.site_import = 1;

    const char *homeCString = [[self pythonHomePath] UTF8String];
    PyStatus status = PyConfig_SetBytesString(&config, &config.home, homeCString);
    if (PyStatus_Exception(status)) {
        if (error) {
            *error = [self errorFromStatus:status];
        }
        PyConfig_Clear(&config);
        return NO;
    }

    const char *programNameCString = [[self programName] UTF8String];
    status = PyConfig_SetBytesString(&config, &config.program_name, programNameCString);
    if (PyStatus_Exception(status)) {
        if (error) {
            *error = [self errorFromStatus:status];
        }
        PyConfig_Clear(&config);
        return NO;
    }

    const char *entryScriptCString = [entryScript fileSystemRepresentation];
    status = PyConfig_SetBytesString(&config, &config.run_filename, entryScriptCString);
    if (PyStatus_Exception(status)) {
        if (error) {
            *error = [self errorFromStatus:status];
        }
        PyConfig_Clear(&config);
        return NO;
    }

    char *argv[] = {(char *)programNameCString, (char *)entryScriptCString};
    status = PyConfig_SetBytesArgv(&config, 2, argv);
    if (PyStatus_Exception(status)) {
        if (error) {
            *error = [self errorFromStatus:status];
        }
        PyConfig_Clear(&config);
        return NO;
    }

    status = Py_InitializeFromConfig(&config);
    PyConfig_Clear(&config);
    if (PyStatus_Exception(status)) {
        if (error) {
            *error = [self errorFromStatus:status];
        }
        return NO;
    }

    [self updateWorkingDirectory];
    return YES;
}

- (void)runEntryScriptAtPath:(NSString *)entryScript {
    const char *scriptPath = [entryScript fileSystemRepresentation];
    FILE *file = fopen(scriptPath, "r");
    if (!file) {
        NSLog(@"[BackendPythonRunner] Unable to open entry script at %s", scriptPath);
        Py_FinalizeEx();
        return;
    }

    PyRun_SimpleFileExFlags(file, scriptPath, 0, NULL);
    fclose(file);

    int finalizeStatus = Py_FinalizeEx();
    if (finalizeStatus != 0) {
        NSLog(@"[BackendPythonRunner] Py_FinalizeEx exited with status %d", finalizeStatus);
    }
}

- (void)updateWorkingDirectory {
    const char *resourcePath = [_resourcePath fileSystemRepresentation];
    if (resourcePath != NULL) {
        chdir(resourcePath);
    }
}

- (NSString *)pythonHomePath {
    return [_resourcePath stringByAppendingPathComponent:@"python"];
}

- (NSString *)programName {
    return @"GoobustersBackend";
}

- (NSError *)errorWithMessage:(NSString *)message {
    return [NSError errorWithDomain:BackendPythonRunnerErrorDomain
                               code:BackendPythonRunnerErrorCodeInitialization
                           userInfo:@{ NSLocalizedDescriptionKey: message ?: @"Unknown error" }];
}

- (NSError *)errorFromStatus:(PyStatus)status {
    NSString *message = status.err_msg ? [NSString stringWithUTF8String:status.err_msg] : @"Unknown Python error";
    return [self errorWithMessage:message];
}

@end
 