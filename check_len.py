import csv, glob
names = []
for p in glob.glob("data/thutuchanhchinh/TTHC_IDs/*/id_tthc.csv"):
    with open(p, 'r') as f:
        for r in csv.DictReader(f):
            names.append(r['PROCEDURE_NAME'])
names.sort(key=len)
print(names[:10])
