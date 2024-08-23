# A [Lip](https://githab.com/liplang/liplang) interpreter inspired by Peter Norvig's lispy:
# - https://github.com/fluentpython/lispy/blob/main/original/norvig/lis.py
# - https://github.com/fluentpython/lispy/blob/main/original/norvig/lispy.py

import re, sys, math
import numpy as np
import operator as op
from copy import copy # deepcopy as dopy(use dopy was a mistake!)
from collections import ChainMap
from functools import reduce as py_reduce

sys.setrecursionlimit(1000000) # 最大递归层数
rsv_words = ['case', 'let', 'quote', 'lambda', 'block', 'release', 'while', 'load', 'print',]
global_env = ChainMap() # (list of dicts, faster and easier)

class ReleaseException(Exception):
    def __init__(self, value):
        self.value = value

#  · <=>---<=> · <=>---<=> · <=>---<=> · <=>---<=> · <=>---<=> · <=>---<=>

def tokenize(expr):
    "将字符串切为令牌(tokens)列表"
    expr = re.sub(r'#.*', '', '(block '+expr+')', flags=re.MULTILINE)  # 删除注释(#)和它的行后
    return (
        expr.replace('{', '(quote (').replace('}', '))').replace('[', '(quote ').replace(']', ')')
            .replace(':', ' ').replace(',', ' ').replace(';', ' ')
            .replace('(', ' ( ').replace(')', ' ) ').replace('\t', ' ').replace('\n', ' ')
            .split()
        )

def lip_parse(program):
    "**解析器** (Parser): 将输入的字符串解析为内部的表达式表示"
    tokenlist = tokenize(program)
    return pack_tokens(tokenlist)

def pack_tokens(tokens):
    "Read an expression from a sequence of tokens."
    if len(tokens) == 0: return []  # raise SyntaxError('unexpected EOF while reading')
    #
    token = tokens.pop(0)
    if token == '(':
        L = []
        while tokens[0] != ')': L.append(pack_tokens(tokens))
        tokens.pop(0)  # pop off ')'
        return L
    elif token == ')':
        raise SyntaxError('unexpected )')
    else:
        return atom(token)

def atom(token):
    'Numbers; booleans; string; None; otherwise Symbol.'
    if token in ['True', 'False', 'None']: return eval(token) # 看成字面常量, 类似与3, 0.18这种
    #
    try: return int(token)
    except:
        try: return float(token)
        except:
            return token  # str(token)?

def unlexer(exp):
    "Convert a Python object back into a Lip-readable string to print ( RE*P*L )."
    if isinstance(exp, list):
        return '(' + ' '.join(map(unlexer, exp)) + ')'
    else:
        return str(exp)

#  · <=>---<=> · <=>---<=> · <=>---<=> · <=>---<=> · <=>---<=> · <=>---<=>

def do_islist (x, env=global_env): return isinstance(x, list)
def do_isint  (x, env=global_env): return isinstance(x, int)
def do_isfloat(x, env=global_env): return isinstance(x, float)
def do_isbool (x, env=global_env): return isinstance(x, bool)
def do_isnone (x, env=global_env): return isinstance(x, type(None))
def do_isnum  (x, env=global_env): return isinstance(x, int) or isinstance(x, float)
def do_isfunc (x, env=global_env): return callable(x)

def do_isnull (x, env=global_env):
    return None if not isinstance(x, list) else len(x) == 0 # 不是列表, 返回None

def do_nop(env): return

def do_list(*exps):
    return list(exps[:-1]) # env = exps[-1]

def do_cons(*exps):
    rtn = [] # env = exps[-1]
    for x in exps[:-1]: rtn += x if isinstance(x, list) else [x]
    return rtn

def do_eval(x, env):
    "Evaluate a Lip expression in a given environment."
    return lip_eval(x, env)

def do_range(low, high, env): # 不用默认值, 因为那样会运行用户写错误代码!
    return list(range(low, high))

def do_car(the_list, env):
    'get the heading item'
    return the_list[0] if the_list else None

def do_cdr(the_list, env):
    'get the tailing sublist (except for the heading item)'
    return the_list[1:]

def do_get(pos, the_list, env):
    'get the item at given positon'
    if pos > len(the_list): raise IndexError("Position out of bounds")
    return the_list[pos]

def do_find(elem, the_list, env):
    # 从列表查找
    if elem in the_list: return the_list.index(elem)
    else: return None  # glacing candy

def do_insert(pos, elem, the_list, env):
    if pos > len(the_list): raise IndexError("Position out of bounds") # 确保位置是合理的
    # 插入元素::
    inlist = [elem] # elem if isinstance(elem, list) else [elem] # 遇到列表直接破开? 可能有问题...
    new_list = the_list[:pos] + inlist + the_list[pos:]
    return new_list

def do_delete(pos, the_list, env):
    if pos >= len(the_list): raise IndexError("Position out of bounds") # 确保位置是合理的
    # 删除元素 elem = the_list[pos]
    new_list = the_list[:pos] + the_list[pos+1:]
    return new_list

def make_lambda(parms, body, env): # make_...
    """创建一个lambda函数，捕获定义时的环境"""
    def lambda_function(*args):
        local_env = copy(env)  # 拷贝外部环境; 别deepcopy! 否则列表l不变: ((lambda x (let l (insert 2 x l))) 6)
        couples = dict(zip(parms, args))
        local_env.update(couples) # 更新局部变量
        return lip_eval(body, local_env)  # 评估函数体
        # 不在lambda中直接引入release...
        # try:
        #     return lip_eval(body, local_env)  # 评估函数体
        # except ReleaseException as e:
        #     return e.value  # 捕获 ReleaseException 并返回其值
    return lambda_function

def do_apply(proc, args, env):
    "Apply a Lip procedure onto the given args."
    if callable(proc):
        return proc(*args)
    else:
        raise TypeError("Not a callable procedure")

def do_map(proc, lst, env): # 使用map：当操作是纯函数、无副作用，而且需要生成新列表时。
    return [proc(x) for x in lst]

def do_reduce(proc, lst, env):
    return py_reduce(proc, lst)

def do_filter(proc, lst, env):
    return [x for x in lst if proc(x)]

inner_defined = {k:vars(math)[k] for k in vars(math) if not k.startswith('__') and k not in ['pow', 'remainder']} # 去掉重复的

pre_defined = {
    '+':op.add, '-':op.sub, '*':op.mul, '/':op.truediv, '**':op.pow, '//': op.floordiv, '%%':op.mod,
    '>':op.gt, '<':op.lt, '>=':op.ge, '<=':op.le, '==':op.eq, '!=':op.ne, '!!':op.not_, '&&':op.and_, '||':op.or_,
    'abs':op.abs, 'neg':op.neg, 'len':op.length_hint, 'random':lambda x=1: list(np.random.random(x)),
    'mean':np.mean, 'std':np.std, 'max': np.max, 'min': np.min, 'euler':np.euler_gamma,
}

lip_defined = {
    'islist':do_islist, 'isint':do_isint, 'isfloat':do_isfloat, 'isnum':do_isnum,
    'isbool':do_isbool, 'isnone':do_isnone, 'isfunc':do_isfunc, 'isnull':do_isnull,
    'nop':do_nop, 'car':do_car, 'cdr':do_cdr, 'list':do_list, 'cons':do_cons, 'eval':do_eval,
    'range':do_range, 'get': do_get, 'find':do_find, 'insert':do_insert, 'delete':do_delete,
    'apply':do_apply, 'map':do_map, 'reduce':do_reduce, 'filter':do_filter,
    'alpha':0.00729735256, '_':[], # 'nil':[],
} # alpha(~ 1/137.036) was not defined in math/operator/numpy...

global_env.update(inner_defined)
global_env.update(pre_defined)
global_env.update(lip_defined)
lip_predef_num = len(global_env)  # 这个数字需要固定下来

#  · <=>---<=> · <=>---<=> · <=>---<=> · <=>---<=> · <=>---<=> · <=>---<=>

def lip_eval(x, env=global_env):
    "**求值器** (Evaluator): 根据环境, 对表达式进行求值"
    if isinstance(x, list):
        if len(x) == 0: return # not x
        else: opper = x[0]
        if   opper == 'case':  # 单独写成函数的话, 不想访问的列表元素也被eval了
            (_, pos, *tobe_access) = x
            pos = lip_eval(pos, env)
            if pos is True:    the_pos =  +1
            elif pos is False: the_pos =   0
            elif pos is None:  the_pos =  -1
            elif isinstance(pos, list):
                if pos:        the_pos =  +1
                else:          the_pos =   0  # 空表为0!
            elif callable(pos):
                raise ValueError("Can determine the case of a func!")
            else:              the_pos = pos
            #
            # the_list = lip_eval(the_list, env) # 错! 不取的, 我们先不去执行!!
            if isinstance(tobe_access, str): tobe_access = env[tobe_access]  # it is the list name? process it!
            if the_pos >= len(tobe_access): raise IndexError("Position out of bounds")
            return lip_eval(tobe_access[the_pos], env)
        elif opper == 'while':  # (while test body)
            (_, test, body) = x
            while lip_eval(test, env): lip_eval(body, env)
            return None  # no returns for loops
        elif opper == 'let':  # 单独写成函数的话, var会被eval
            (_, var, exp) = x
            the_exp = lip_eval(exp, env) # why,, really need?
            # if str(var) in rsv_words: raise ValueError("Can not assign to a keyword!")
            if type(var) in [int, float, bool, type(None)]: raise ValueError("Can not assign to a const!")
            if var: env[var] = the_exp
        elif opper == 'quote': # (quote exp)  # 单独写成函数的话, exp总会被eval
            (_, exp) = x
            return exp # Lispy languages do NOT proc ops and ids here!!!
        elif opper == 'lambda': # (lambda parms func)
            (_, parms, body) = x # 定义成单独函数的话, 其临时变量(参数)会被错误求值(eval)
            return make_lambda(parms, body, env)
        elif opper == 'block': # (block (expr) ...); (block (let r 10) (let pi 3.14) (* pi (* r 2))) # 6.28
            try:
                block_value = None
                for sub_expr in x[1:]: block_value = lip_eval(sub_expr, env)
                return block_value
            except ReleaseException as e:
                return e.value
        elif opper == 'release': # (release exp) in `block` only!!
            (_, exp) = x
            result = lip_eval(exp, env)
            raise ReleaseException(result)  # 使用异常来传递返回值
        elif opper == 'load': # (load filename)
            (_, filename) = x
            with open(filename, 'r', encoding='utf-8') as file: program = file.read()
            expr = lip_parse(program)
            lip_eval(expr, env)
        elif opper == 'print': # (print expr)
            (_, expr) = x
            the_expr = lip_eval(expr)
            print(unlexer(the_expr))
        else:  # (proc arg...) (函数名 参数1 参数2 ...)
            proc = lip_eval(opper, env)
            args = [lip_eval(arg, env) for arg in x[1:]]
            if callable(proc) and proc.__name__.startswith('do_'): args.append(env)
            # opper in lip_defined会出错, it maybe list => unhashable!
            return proc(*args)
    # 不是列表的话, 那就是 `变量` 或 `字面常量`:
    elif isinstance(x, str):
        return env[x]  # 获取环境中的函数或变量
    else:
        return x  # 数字等字面常量直接返回

#  · <=>---<=> · <=>---<=> · <=>---<=> · <=>---<=> · <=>---<=> · <=>---<=>

def help(): # do_help(env=global_env)? 不用默认值, 因为那样会运行错误代码: (help 1)
    print(f'\nREPL commands: `help` -> get help, `clear` -> clear screen, `exit/quit` -> exit.')
    print(f'\nThe language has\n\t{len(rsv_words)} Reserved Words: {rsv_words},\n\t{6} Data Types: int, float, bool, None, func, list,')
    print(f'\tas well as {lip_predef_num} Predefines in initial.')
    print(f'\nIt has {len(global_env)} defines in total now: {list(global_env.keys())}.')
    print('\nEye Candy: [a] => (quote a); {x y ...} => (quote (x y ...)); characters like ,:; => spaces.')

def lip_repl(prompt='lip> ', remind=''):
    "Read-Eval-Print: 循环持续地读取用户输入, 解析, 求值, 打印结果, 直到用户手动终止程序"
    print('\n\nWelcome to LIP programming 1.00!')
    help()

    while True:
        print()
        try:
            sc = input(prompt).strip() # "read ; [R]EPL loop"  只能读入单行!!
            if   len(sc) == 0: continue  # 注意次序! 空语句, 直跳过
            elif sc[0] == '#': continue  # 注意次序! 空语句, 直跳过
            elif sc in ['\x04', '\x05', 'exit', 'quit']: return  # ^D|^E to exit;
            elif sc == 'help': help()
            elif sc == 'clear': print('\n'*36)
            else:
                tc = lip_parse(sc)
                val = lip_eval(tc)
                if val is not None: print(remind + unlexer(val))
        except Exception as e:
            print(f'[Error] {type(e).__name__}: {e}')


if __name__ == '__main__': lip_repl()
