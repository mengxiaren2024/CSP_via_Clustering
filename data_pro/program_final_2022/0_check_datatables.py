import os
for i in range(1,100):
    if(i<51):
      filename="/Volumes/Elements/NEW_CSP_DATA/TOP-"+str(i)+"000/"
    else:
      filename = "/Volumes/Elements2/NEW_CSP_DATA/TOP-" + str(i) + "000/"
    csp_header_file=filename+"csp_header_nov_"+str(i)+".json"
    logs=filename+"logs_nov_"+str(i)+".json"
    if not (os.path.exists(csp_header_file)):
        print(str(i),": missing ",csp_header_file)
    if not (os.path.exists(logs)):
        print(str(i), ": missing ", logs)
    if not (os.path.exists(filename+"Status_Code_1.csv")):
        print(str(i), ": missing Status_Code_1.csv")
    if not (os.path.exists(filename+"CSP_Meta_1.csv")):
        print(str(i), ": missing CSP_Meta_1.csv")
    if not (os.path.exists(filename+"Html_1.csv")):
        print(str(i), ": missing Html_1.csv")
    if not (os.path.exists(filename+'S_RL_1.csv')):
        print(str(i), ": missing S_RL_1.csv")
    if not(os.path.exists(filename+'Request_1.csv')):
        print(str(i), ": missing Request_1.csv")

array=[52, 55, 64, 65, 66, 67, 68, 69, 70, 73, 74, 75, 86, 89, 11, 31, 34, 38, 46]
for i in array:
    if(i<51):
      filename="/Volumes/Elements/NEW_CSP_DATA/TOP-"+str(i)+"000-2/"
    else:
      filename = "/Volumes/Elements2/NEW_CSP_DATA/TOP-" + str(i) + "000-2/"
    csp_header_file=filename+"csp_header_nov_"+str(i)+"_2.json"
    logs=filename+"logs_nov_"+str(i)+"_2.json"
    if not (os.path.exists(csp_header_file)):
        print(str(i),": missing ",csp_header_file)
    if not (os.path.exists(logs)):
        print(str(i), ": missing ", logs)
    if not (os.path.exists(filename+"Status_Code_1.csv")):
        print(str(i), ": missing Status_Code_1.csv")
    if not (os.path.exists(filename+"CSP_Meta_1.csv")):
        print(str(i), ": missing Status_Code_1.csv")
    if not (os.path.exists(filename+"Html_1.csv")):
        print(str(i), ": missing Html_1.csv")
    if not (os.path.exists(filename+'S_RL_1.csv')):
        print(str(i), ": missing S_RL_1.csv")
    if not(os.path.exists(filename+'Request_1.csv')):
        print(str(i), ": missing Request_1.csv")


array2=[ 66, 89]
for i in array2:
    if(i<51):
      filename="/Volumes/Elements/NEW_CSP_DATA/TOP-"+str(i)+"000-3/"
    else:
      filename = "/Volumes/Elements2/NEW_CSP_DATA/TOP-" + str(i) + "000-3/"
    csp_header_file=filename+"csp_header_nov_"+str(i)+"_3.json"
    logs=filename+"logs_nov_"+str(i)+"_3.json"
    if not (os.path.exists(csp_header_file)):
        print(str(i),": missing ",csp_header_file)
    if not (os.path.exists(logs)):
        print(str(i), ": missing ", logs)
    if not (os.path.exists(filename+"Status_Code_1.csv")):
        print(str(i), ": missing Status_Code_1.csv")
    if not (os.path.exists(filename+"CSP_Meta_1.csv")):
        print(str(i), ": missing Status_Code_1.csv")
    if not (os.path.exists(filename+"Html_1.csv")):
        print(str(i), ": missing Html_1.csv")
    if not (os.path.exists(filename+'S_RL_1.csv')):
        print(str(i), ": missing S_RL_1.csv")
    if not(os.path.exists(filename+'Request_1.csv')):
        print(str(i), ": missing Request_1.csv")