import pefile
pe = pefile.PE(r".\\.venv\\Lib\\site-packages\\diffsol_pytorch\\diffsol_pytorch.cp313-win_amd64.pyd")
imports = sorted({entry.dll.decode() for entry in pe.DIRECTORY_ENTRY_IMPORT})
print("Imports:")
for name in imports:
    print(" -", name)
