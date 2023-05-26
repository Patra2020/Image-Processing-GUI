

def open_ImageJ():
    import subprocess, os, platform, glob
    filepath = "ImageJ\ImageJ.exe"
    list_of_images = glob.glob("images/*.tiff")
    subprocess.Popen([filepath,list_of_images[len(list_of_images) - 1]])

