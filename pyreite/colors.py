"""ANSI color helpers for terminal output."""


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def printred(string):
    print(f"{bcolors.FAIL}{string}{bcolors.ENDC}")

def printyellow(string):
    print(f"{bcolors.WARNING}{string}{bcolors.ENDC}")

def printgreen(string):
    print(f"{bcolors.OKGREEN}{string}{bcolors.ENDC}")

def printblue(string):
    print(f"{bcolors.OKBLUE}{string}{bcolors.ENDC}")
