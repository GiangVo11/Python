
# coding: utf-8

# In[ ]:


import csv
from datetime import datetime

output_file = open(r"F:\Giang\corrected.csv", "wb")
fieldnames = ['Date', 'Month' , 'Year', 'Hour', 'AQI' , 'Raw Conc.']
writer = csv.DictWriter(output_file, fieldnames=fieldnames)
writer.writeheader()

with open(r"F:\Giang\selected.csv") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        output_row = {}
        change_date = datetime.strptime(row['Date (LT)'], '%d-%m-%Y %H:%M')
        output_row['Date'] = change_date.strftime('%d')
        output_row['Month'] = change_date.strftime('%B')
        output_row['Year'] = change_date.strftime('%Y')
        output_row['Hour'] = change_date.strftime('%I %p')
        output_row['AQI'] = row['AQI']
        output_row['Raw Conc.'] = row['Raw Conc.']
        writer.writerow(output_row)

output_file.close()

