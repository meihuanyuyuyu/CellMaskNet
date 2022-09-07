from config import arg

print(dir(arg))
for _ in dir(arg):
    if not _.startswith('_'):
        print(_,'=',getattr(arg,__package__))
