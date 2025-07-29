---
date: 2025-05-27
author:
  - Siyuan Liu
tags:
  - FIT9136
aliases:
  - quiz3
---
# Week 9
## Prologue
事实上quiz的重点不会放在概念的理解上面，但是继承的概念是所有涉及OOP的高级计算机语言中非常重要的一环, 在此不多赘述，详情请看ED的描述([ED Definition of Inheritance](https://edstem.org/au/courses/20671/lessons/72624/slides/484409)).

## How Inheritance Happen?
**如下是常见继承的示例**
```Python
class Animal:
	def eat():
		pass


class Cat(Animal):
	def eat(food):
		print(f"eat {food}")
	
```
我们能够很清楚的看到，假如定义类的旁边多了一个括号，就代表这个类发生了继承，继承了==括号内的那个类==.
简而言之，定义类时，类名后面有括号，说明该类是子类，括号里的类是父类.

## How Inheritance Works?
现实生活中的继承一般就是直系亲属的资产过继给后代， **Python** 的继承也是类似的，它会把父类定义的所有 **variables** 和 **methods** **“过继”** 给子类.
==Attention:==
1. ==在使用从父类 **过继** 过来的 **variables** 和 **methods** 时,必须通过 **super().** 访问.==
2. ==在子类的 **__init__()** method的入参中,必须包含父类 **__init__()** 的所有入参,并且顺序一致.==
3. 
## What Is super()?
当我们谈起 **super()** 其实就绕不开 **self** ，他们会让你有一种类似的感觉，特别是在 **JAVA** 里面 -> **super** 与 **this**.
**self**  是Python 方法的第一个参数`(如果你的function/method需要这个参数的话)`，代表调用该方法的对象本身.

而 **super()** 相对于返回了一个父类对象，通过 **.** 这个符号来调用父类存在的变量与method.
**如下是常见super()的示例**
```Python
class Animal:
	def __init__(self, weight):
		self.weight = weight

	def eat(self):
		pass


class Cat(Animal):
	def __init__(self, weight, color):
		super().__init__(weight)
		self.color = color
		
	def eat(self, food):
		print(f"eat {food}")
```

## Abstract Classes
## How Abstract Classes Happen?
**如下是常见继承的示例**
```Python
from abc import ABC, abstractmethod


class Animal(ABC):
	@abstractmethod
	def eat(self):
		pass


class Cat(Animal):
	def eat(self, food):
		print(f"eat {food}")
	
```
我们能够轻松发现,只要一个类继承了 **ABC** 这个抽象基类,那么这个类就叫做抽象类, 抽象类不应该被实例化`(虽然你实例化一个没有抽象method的抽象类并不会报错)`, 但是切记避免实例化抽象类, 它只应该被拿来继承, 那么抽象类有什么特别之处呢? 答案就是 -> 定义抽象method.

## What Is Abstract Method?
从上面的示例可以看到,头上戴了一个 **@abstractmethod** 的方法就是抽象method, 抽象method是只能在抽象类中定义的方法, 它的特点在于, 如果子类继承了含有抽象method的父类,那么这个抽象method必须被override. 而且抽象method一般不应该有任何业务逻辑`(也就是pass)`. 
==Attention:==
1. ==如果你实例化了一个有抽象method的抽象类, 那么IDE会抛出TypeError.==
2. ==如果你继承了一个拥有抽象method的抽象类, 并且没有override抽象method, 那么IDE会抛出TypeError.==

----------------------------==因为时间不多了所以之后直接写考点==---------------------------

# Week 9
1. 通过 **obj.variable/obj.metthod** 调用method/variable时, interpreter 在查找method/variable名字的时候,是按照: 
	`检查实例: 实例对象是否有该method/variable? 如果有, 使用它.`
	`检查类：在对象的类中查找method/variable`
	`检查父类: 在对象的父类中查找variable或method`
	`如果以上环节都没有找到,则抛出错误错误`
	
**question 1:**
```Python
class Animal:  
    def __init__(self, weight):  
        self.weight = weight  
  
    def eat(self):  
        print("eat")  
  
    def sleep(self):  
        print("sleep")  
  
  
class Cat(Animal):  
    def __init__(self, weight, color):  
        super().__init__(weight)  
        self.color = color  
  
    def eat(self):  
        print(f"eat fish")  
  
    def jump(self):  
        print("cat jump")  
  
if __name__ == "__main__":  
    animal = Animal(20)  
    cat = Cat(10, "yellow")  
  
    animal.eat()  
    animal.sleep()  
    cat.eat()  
    cat.sleep()  
    cat.jump()
    print(animal.weight)
    print(cat.weight)
    animal.jump()
```
**what display on console:**
```spoiler-block
IDE will throw out an AttributeError: 'Animal' object has no attribute 'jump'.

If we remove 'animal.jump()', then console will display:

eat
sleep
eat fish
sleep
cat jump
20
10
```

2. Python原生不支持overload, 所以同名function/method不同的入参,会导致后定义的会覆盖前面的.

**question 1:**
```Python
class Animal:  
    def __init__(self, weight):  
        self.weight = weight  
  
    def eat(self):  
        print("eat")  
  
  
class Cat(Animal):  
    def __init__(self, weight, color):  
        super().__init__(weight)  
        self.color = color  
  
    def eat(self, food):  
        print(f"eat {food}")  

  
if __name__ == "__main__":  
    animal = Animal(20)  
    cat = Cat(10, "yellow")  
  
    animal.eat()  
    cat.eat("fish")  
```
**what display on console:**
```spoiler-block
eat
eat fish
```

3. 抽象类不应该被实例化`(虽然你实例化一个没有抽象method的抽象类并不会报错)`, 但是切记避免实例化抽象类, 它只应该被拿来继承

**question 1:**
```Python
from abc import ABC, abstractmethod  
  
  
class Animal(ABC):  
    def eat(self):  
       print(f"eat")  
  
  
class Cat(Animal):  
    def eat(self, food):  
       print(f"eat {food}")  
  
  
if __name__ == "__main__":    
    animal = Animal()    
	cat = Cat()  
    animal.eat()  
    cat.eat("fish") 
```
**what display on console:**
```spoiler-block
eat
eat fish
```

4. 抽象方法一定要在子类override

**question 1:**
```Python
from abc import ABC, abstractmethod  
  
  
class Animal(ABC):  
    @abstractmethod  
    def eat(self):  
        print(f"eat")  
  
  
class Cat(Animal):  
    def jump(self):  
        print("cat jump")  
  
  
if __name__ == "__main__":  
    animal = Animal()  
    cat = Cat()  
    animal.eat()  
    cat.jump()
```
**what display on console:**
```spoiler-block
TypeError: Can't instantiate abstract class Animal without an implementation for abstract method 'eat'
```

5. 子类的构造函数的入参必须包含父类所有的入参,顺序也需要一致.

**question 1:**
```Python
class Animal:  
    def __init__(self, weight, color):  
        self.weight = weight  
        self.color = color  
  
    def eat(self):  
        pass  
  
class Cat(Animal):  
    def __init__(self, breed, color, weight):  
        super().__init__(color,weight)  
        self.breed = breed  
  
if __name__ == "__main__":  
    animal = Animal(10, "white")  
    cat = Cat(20, "British Shorthair", "blue")  
    print(cat.breed)  
    print(cat.color)  
    print(cat.weight)
```
**what display on console:**
```spoiler-block
20
blue
British Shorthair

Attention:
1. 如果入参是乱序的,但是参数之间类型不同,那么intepreter会自动匹配入参,这也是为什么print(cat.color) 和 print(cat.weight)是之前的答案,而不是"British Shorthair"和"blue"
```

# Week 10
1. try-except-else-finally的运行顺序是:
- **try 块**
- 解释器先执行 `try` 代码块中的内容。
- 如果没有发生异常，则继续执行 `else` 块（如果有）。
- 如果发生异常，则立即跳到对应的 `except` 块处理。
- **except 块**
- 只有在 `try` 代码块中发生了“匹配的异常”时，才会执行 `except` 块。
- 异常处理完后，跳过 `else` 块，执行 `finally` 块（如果有）。
- **else 块**
- 只有 `try` 代码块**没有发生任何异常**时，才会执行 `else` 块。
- 如果 `try` 里发生异常，则 `else` 块不会被执行。
 - **finally 块**
- `finally` 块**总是会被执行**，无论有没有发生异常、异常是否被捕获，甚至在 `try` 或 `except` 块中有 `return`、`break`、`continue`，`finally` 都会执行。

**question 1:**
```Python
try:
    print("try")
except ValueError:
    print("except")
else:
    print("else")
finally:
    print("finally")
```
**what display on console:**
```spoiler-block
try
else
finally
```

**question 2:**
```Python
try:
    raise ValueError("error")
except ValueError:
    print("except")
else:
    print("else")
finally:
    print("finally")
```
**what display on console:**
```spoiler-block
try
except
finally
```

2. 异常的捕获也是按先后顺序来的

**question 1:**
```Python
try:
    nums = [1, 2, 3]
    print(nums[3])
    result = 10 / 0
except ZeroDivisionError:
    print("Division error")
except IndexError:
    print("Invalid index")
finally:
    print("Cleanup done")
```
**what display on console:**
```spoiler-block
Invalid index
Cleanup done
```

3. 在try块中发生的数据操作都是永久性的

**question 1:**
```Python
lst = ["2", "3", "5", "7", "nine", "11"]
total = 0
for s in lst:
    try:
        total += int(s)
    except ValueError:
        pass
print(total)
```
**what display on console:**
```spoiler-block
28
```

4. 每个测试method都应使用以 `test` 为前缀的名字。此命名约定使 unittest 测试运行器能够自动识别测试方法并执行它们

5. 常见边际情况:
![[Pasted image 20250527200556.png]]

6. 常见错误
- **SyntaxError:** 这类错误是由于程序的语法错误而发生的。例如， `if-statement` 后缺少一个冒号 `(:)` 就是语法错误（也称为解析错误）

- **NameError:** 当您尝试在初始化之前使用某个值, 尝试在未先导入的情况下使用某个包时，就会发生此类错误

- **TypeError:** 当您尝试在单个语句中使用不兼容的数据类型或向函数传递错误类型的参数时，就会发生此类错误。例如，将字符串和整数相加就是类型错误的一个例子；将列表传递给期望整数参数的函数也是类型错误的另一个例子。

- **ValueError:** 当函数接收的参数数据类型正确，但值不合适时，就会发生此类错误。例如，向内置函数 math.sqrt `.sqrt()` 传递负数将导致值错误。

5. 注意编写测试类的时候一定要继承`unittest.TestCase`

6. 
```Python
import unittest subtraction_function = lambda x, y: x - y class TestSubtraction(): def test_subtraction(self): self.assertEqual(subtraction_function(8,3),7) if __name__ == "__main__": unittest.main()
```
**what display on console:**
```spoiler-block
什么都不会发生,因为TestSubtraction()根本不是测试类
```

# Week11
1. 
- A recursive algorithm must have a **base case**.
- A recursive algorithm must change its state and move toward the base case (**convergence condition**).
- A recursive algorithm must call itself, recursively.

1. Fibonacci sequence有两种类型0 and 1开头或1 and 1开头,所以有关斐波那契数列的递归, **base case** 应该是
```Python
 def factorial(n): 
    if n == 1or n == 0: 
        return 1 
    else: 
        return n * factorial(n-1) 
```

2. Converting an Integer to a String in Any Base
自行理解该递归代码
- Tips: `leadingDigits = toStr(n//base, base)`是为了从最高位逐个取值
- `lastDigit = convertString[n%base]`是为了对无法除尽的最后一位进行匹配值
```Python
def toStr(n, base):  
    convertString = "0123456789ABCDEF"  
    if n < base:  
        return convertString[n]  
    else:  
        leadingDigits = toStr(n//base, base)  
        lastDigit = convertString[n%base]  
        return leadingDigits + lastDigit  
  
print(toStr(1453, 16))
```