#!/usr/bin/env python3
"""
Entrypoint for Goobusters tracking tools.
Loads settings from dot.env and performs tracking inline.
"""
import runpy

def main():
    # Execute the lib.optical script as __main__, loading DEBUG, dot.env, and running tracking
    runpy.run_module('lib.optical', run_name='__main__')

if __name__ == "__main__":
    main()