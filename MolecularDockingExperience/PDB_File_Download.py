from selenium import webdriver
from selenium.webdriver.edge.service import Service
import time

driver_path = r'C:\Program Files (x86)\Microsoft\Edge\Application\edgedriver_win64\msedgedriver.exe' # Location of the drive.
prefs = {'profile.default_content_settings.popups': 0, 
         'download.default_directory': r'C:\......\MolecularDockingExperience\receptor'} # Set the storage path for the downloaded file, here you need the absolute path.
service = Service(executable_path = driver_path)
options = webdriver.EdgeOptions()
options.add_experimental_option('prefs', prefs)
options.add_argument('headless') # Browser background launch.
driver = webdriver.Edge(service = service, options = options)

with open("PDB_ID.txt", "r") as f: # Open file.
    data = f.read().split(' ') # Read file.
    for i in range(len(data)):
        url = "https://files.rcsb.org/download/" + data[i] + '.pdb'
        driver.get(url)
        time.sleep(1)