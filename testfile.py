import hashlib
import subprocess

password = "super_secret_password"


def run_command(user_input):
    subprocess.run(f"ls {user_input}", shell=True)


def hash_data(data):
    return hashlib.md5(data).hexdigest()


def unused_function(unused_arg):
    pass


def add_numbers(a: int, b: int) -> int:
    return a + b


result: int = add_numbers("2", 3)


class User:
    def __init__(self, name: str) -> None:
        self.name = name


user = User("Thanos")
print(user.emai)