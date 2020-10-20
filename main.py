"""
main.py

@time: 10/20/20
@author: Qinxin Wang

@desc:
"""

from models import composer

if __name__ == "__main__":
	jazzGen = composer.Generator()
	jazzGen.generate()