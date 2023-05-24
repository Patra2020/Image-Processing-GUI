

def open_ImageJ():
    import subprocess, os, platform
    filepath = "ImageJ\ImageJ.exe"
    # if platform.system() == 'Darwin':       # macOS
    #     subprocess.call(('open', filepath))
    # elif platform.system() == 'Windows':    # Windows
    #     os.startfile(filepath)
    # else:                                   # linux variants
    #     subprocess.call(('xdg-open', filepath))
    subprocess.Popen([filepath,"hi.jpg"])

