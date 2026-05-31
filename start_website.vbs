Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
base = fso.GetParentFolderName(WScript.ScriptFullName)
cmd = """" & base & "\.venv\Scripts\python.exe"" """ & base & "\website.py"" > """ & base & "\website.log"" 2>&1"
shell.CurrentDirectory = base
shell.Run "%ComSpec% /c " & cmd, 0, False
