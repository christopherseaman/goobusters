#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface BackendPythonRunner : NSObject

- (instancetype)initWithResourcePath:(NSString *)resourcePath;
- (BOOL)startWithEntryScript:(NSString *)entryScript error:(NSError **)error NS_SWIFT_NAME(start(withEntryScript:));
- (void)stop;
- (BOOL)isRunning;

@end

NS_ASSUME_NONNULL_END
