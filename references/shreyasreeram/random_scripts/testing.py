from dropbox import Dropbox
import os

dbx = Dropbox(os.getenv("DROPBOX_ACCESS_TOKEN"))

for entry in dbx.files_list_folder("").entries:
    print(entry.name)