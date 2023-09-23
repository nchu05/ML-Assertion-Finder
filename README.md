# ML-Assertion-Finder
This project was done in collaboration with Cornell CS Professor Saikat Dutta (Ph.D.). 

Many machine learning (ML) algorithms are inherently random – multiple executions using the same inputs may produce slightly different 
results each time. Randomness impacts how developers write tests that check for end-to-end quality of their implementations of
these ML algorithms. Often, randomness results in tests that are "flaky" or nondeterministic. This project is a tool to log approximate
 assertion tests in projects that utilize common ML libraries.

To run assertion finder, go to the AssertSpecFinder.py file and, under the main method, change "pathToProjectDirectory" and "projectName" with the appropriate directory and name.

To use static instrumentation tool, edit "testFilePath," "testName," and "lineNumber" using results from the assertion finder. 