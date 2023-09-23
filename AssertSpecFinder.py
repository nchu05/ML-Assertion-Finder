import ast
import csv
import os
import shutil

acceptableAssertions = ('assertAlmostEqual', 'assertGreater', 'assertGreaterEqual', 'assertLess', 'assertLessEqual', 'assert_almost_equal', 'assert_approx_equal', 'assert_array_almost_equal', 'assert_allclose', 'assert_array_less', 'assertAllClose')
compareName = {ast.Lt:'assertLess', ast.LtE:'assertLessEqual', ast.Gt:'assertGreater', ast.GtE:'assertGreaterEqual'}

class Visitor(ast.NodeVisitor):
    def __init__(self):
        self.assertions = []
        self.path = None
        self.testclass = None
        self.testname = None

    def visit_ClassDef(self, node):
        self.testclass = node.name
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        self.testname = node.name
        self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.attr in acceptableAssertions:
            lineNumber = node.lineno
            assertString = ast.unparse(node)
            print("Found call assert: ", node.func.attr, " Line: ", node.lineno)
            self.assertions.append((self.path, self.testclass, self.testname, node.func.attr, lineNumber, assertString)) 
        elif isinstance(node.func, ast.Name) and node.func.id in acceptableAssertions:
            lineNumber = node.lineno
            assertString = ast.unparse(node)
            print("Found call assert: ", node.func.id, " Line: ", lineNumber)
            self.assertions.append((self.path, self.testclass, self.testname, node.func.id, lineNumber, assertString)) 
        self.generic_visit(node)
    
    def visit_Assert(self, node):
        if isinstance(node.test, ast.Compare) and isinstance(node.test.ops[0], (ast.Lt, ast.LtE, ast.Gt, ast.GtE)):
                print("Found assert: ", node.test.ops[0], " Line: ", node.test.lineno)
                assertString = ast.unparse(node)
                lineNumber = node.lineno
                self.assertions.append((self.path, self.testclass, self.testname, compareName[node.test.ops[0].__class__], lineNumber, assertString)) 
        self.generic_visit(node)
    

def writeToCsv(assertions, project):
    csvFilename = f"{project}_assertions.csv"
    with open(csvFilename, 'w') as csvfile:
        csvHandler = csv.writer(csvfile)
        csvHandler.writerows(assertions)
    print("Finished!")   

def transverseFiles(project):
    visitor = Visitor()
    for dirpath, dirnames, filenames in os.walk(project):
        for file in filenames:
            if 'test' in file and '.py' in file:
                filePath = os.path.join(dirpath, file)
                visitor.path = filePath
                print(visitor.path)
                with open(filePath, "r") as current:
                    tree = ast.parse(current.read())
                    visitor.visit(tree) 
    return visitor.assertions

def main():
    assertion = transverseFiles(pathToProjectDirectory)
    writeToCsv(assertion, projectName)
    """
    example:
    assertion = transverseFiles("task2results/allentests")
    writeToCsv(assertion, "allentests")
    """

if __name__ == "__main__":
    main()