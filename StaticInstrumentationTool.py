import ast
import shutil
from ast import *
import astunparse

acceptableAssertions = ('assertAlmostEqual', 'assertGreater', 'assertGreaterEqual', 'assertLess', 'assertLessEqual', 'assert_almost_equal', 'assert_approx_equal', 'assert_array_almost_equal', 'assert_allclose', 'assert_array_less', 'assertAllClose')
assertBoolean = ('assertTrue', 'assertFalse')

importNumpy = ast.Import(names=[ast.alias(name='numpy', asname='np')])
importTorch = ast.Import(names=[ast.alias(name='torch', asname=None)])
importPytest = ast.Import(names=[ast.alias(name='pytest', asname=None)])
importTensorFlow = ast.Import(names=[ast.alias(name='tensorflow', asname='tf')])

inputString = """
def complexObj(string):
    try:
        if isinstance(string, torch.Tensor):
            return np.array2string(string.detach().numpy(), precision=50, separator=',').replace('\\n', '')
    except:
        pass
    if isinstance(string, np.ndarray):
        return np.array2string(string, precision=50, separator=',').replace('\\n', '')
    if isinstance(string, list):
        for i in string:
            return "[{0}]".format(','.join([complexObj(i)]))
    try:
        if isinstance(string, tf.Tensor):
            return np.array2string(string.eval(), precision=50, separator=',').replace('\\n', '')
    except:
        pass
    return str(string)
    """

class staticInstrumentationTool(ast.NodeTransformer): 
    def __init__(self, testFile, testName, assertionLine):
            self.testFile = testFile
            self.testName = testName
            self.assertionLine = assertionLine
            self.imports = []
            self.logString = "log>>>" 
            self.modifiedAst = None
            self.visited = False
            self.val = 1        
            self.func_col_offset = 0
            self.modifiedAstBase = None

    def createImportString(self):
        inputStringAst = ast.parse(inputString)
        return inputStringAst.body[0]
    
    def helper(self):
        with open(self.testFile) as file:
            fileInAst = ast.parse(file.read())
            tree = self.visit(fileInAst)
            tree = ast.fix_missing_locations(tree)
            importString = self.createImportString()
            nodeIndex = [i for i in range(len(tree.body)) if isinstance(tree.body[i], ast.ClassDef)]
            if len(nodeIndex) == 0:
                nodeIndex = [0]
            nodeIndex = nodeIndex[0]
            tree.body.insert(nodeIndex, importString)
            tree.body.insert(nodeIndex, importPytest)
            tree.body.insert(nodeIndex, importNumpy)
            if 'numpy' in self.imports:
                tree.body.insert(nodeIndex, importNumpy)
            if 'torch' in self.imports:
                tree.body.insert(nodeIndex, importTorch)
            if 'tensorFlow' in self.imports:
                tree.body.insert(nodeIndex, importTensorFlow)
            self.modifiedAstBase = tree

        with open(self.testFile) as file:
            fileInAst = ast.parse(file.read())
            self.visit(fileInAst)
            tree = self.visit(fileInAst)
            tree = ast.fix_missing_locations(tree)
            nodeIndex = [i for i in range(len(tree.body)) if isinstance(tree.body[i], ast.ClassDef)]
            if len(nodeIndex) == 0:
                nodeIndex = [0]
            nodeIndex = nodeIndex[0]
            importString = self.createImportString()
            tree.body.insert(nodeIndex, importString)            
            tree.body.insert(nodeIndex, importPytest)            
            tree.body.insert(nodeIndex, importNumpy)
            if 'numpy' in self.imports:
                tree.body.insert(nodeIndex, importNumpy)
            if 'torch' in self.imports:
                tree.body.insert(nodeIndex, importTorch)
            if 'tensorflow' in self.imports:
                tree.body.insert(nodeIndex, importTensorFlow)
            self.modifiedAst = tree

    def writeIntoFile(self):
        shutil.copy(self.testFile, self.testFile + ".bak")
        with open(self.testFile, 'w') as file:
            file.write(astunparse.unparse(self.modifiedAst).strip())
        shutil.copy(self.testFile, self.testFile + ".bak")
        with open(self.testFile, 'w') as file:
            file.write(astunparse.unparse(self.modifiedAstBase).strip())

    def log(self, node):
        assign = Assign(targets=[Name(id='val_{0}'.format(self.val), ctx=Store())], value=node)
        print = ast.parse("print(('{0}%s' % complexObj({1}) ))".format(self.logString, 'val_{0}'.format(self.val)))
        self.val += 1
        return [assign, print]

    def visit(self, node):
        if isinstance(node, ast.Import):
            print(node.names[0].name)
            self.imports.append(node.names[0].name)
        if isinstance(node, ast.Assert):
            if isinstance(node.test, ast.Compare) :
                array = []
                print1 = self.log(node.test.left)
                node.test.left = print1[0].targets[0]
                array.extend(print1)
                for i, j in enumerate(node.test.comparators):
                    print1 = self.log(j)
                    array.extend(print1)
                    node.test.comparators[i] = print1[0].targets[0]
                array.append(node)
                return array
            elif isinstance(node.test, ast.Call):
                array = []
                for i, j in enumerate(node.test.args):
                    if isinstance(j, ast.Name):
                        print1 = self.log(j)
                        node.test.args[i] = print1[0].targets[0]
                        array.extend(print1)
                    elif isinstance(j, ast.Compare):
                        print1 = self.log(j.left)
                        j.left = print1[0].targets[0]
                        array.extend(print1)
                        for ii, jj in enumerate(j.comparators):
                            print1 = self.log(jj)
                            array.extend(print1)
                            c.comparators[ii] = print1[0].targets[0]
                array.append(node)
                return array
        elif isinstance(node, ast.Expr):
            if isinstance(node.value, ast.Call):
                if isinstance(node.value.func, ast.Name):
                    array = []
                    if node.value.func.id in acceptableAssertions:
                        if node.value.func.id in assertBoolean and isinstance(node.value.args[0], ast.Compare):
                            print1 = self.log(node.value.args[0].left)
                            array.extend(print1)
                            node.value.args[0].left = print1[0].targets[0]
                            for i, j in enumerate(node.value.args[0].comparators):
                                print2 = self.log(j)
                                array.extend(print2)
                                node.value.args[0].comparators[i] = print2[0].targets[0]
                        else:
                            for i, j in enumerate(node.value.args):
                                print1 = self.log(j)
                                array.extend(print1)
                                node.value.args[i] = print1[0].targets[0]
                    array.append(node)
                    return array
                elif isinstance(node.value.func, ast.Attribute):
                    array = []
                    if node.value.func.attr in acceptableAssertions :
                        if node.value.func.attr in assertBoolean and isinstance(node.value.args[0],ast.Compare) :
                            print1 = self.log(node.value.args[0].left)
                            node.value.args[0].left = print1[0].targets[0]
                            array.extend(print1)
                            for i, j in enumerate(node.value.args[0].comparators):
                                print2 = self.log(j)
                                array.extend(print2)
                                node.value.args[0].comparators[i] = print2[0].targets[0]
                        else:
                            for i, j in enumerate(node.value.args):
                                print1 = self.log(j)
                                array.extend(print1)
                                node.value.args[i] = print1[0].targets[0]
                    array.append(node)
                    return array
        self.generic_visit(node)
        return node

def main():
    visitor = staticInstrumentationTool(testFilePath, testName, lineNumber)
    visitor.helper()
    visitor.writeIntoFile()
    print("finished!")

if __name__ == "__main__":
    main()
