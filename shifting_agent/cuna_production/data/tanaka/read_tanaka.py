setup = [[0, 240, 25, 265, 30, 270, 35, 275],
         [240, 0, 265, 25, 270, 30, 275, 35],
         [35, 275, 0, 240, 25, 265, 20, 260],
         [275, 35, 240, 0, 265, 25, 260, 20],
         [40, 280, 35, 275, 0, 240, 25, 265],
         [280, 40, 275, 35, 240, 0, 265, 25],
         [45, 285, 40, 280, 35, 275, 0, 240],
         [285, 45, 280, 40, 275, 35, 240, 0]]


class Job:
    jTime = 1
    jDue = 5
    jType = 0

 

    def __init__(self,time,due,jtype):
        self.jTime=time
        self.jDue=due
        self.jType=jtype


def read_tanaka(filename):
    jobs_array = []

 

    with open(filename, "r") as infile:
        reader = csv.reader(infile)

        # Skip the header row
        #next(reader, None)

        counter=0

        for row in reader:
            if len(row) >= 3:
                jTime = int(row[0])*50
                jDue = int(row[1])*50
                jType = int(row[2])

 

                job = LocalSearch.Job(jTime, jDue, jType)
                jobs_array.append(job)
                counter+=1

                #if counter>50:
                #    break

    return jobs_array