import sys
import random

def generate_grid_graph(lx, ly, degree, output_prefix):

    loc_file = open("{}locations.csv".format(output_prefix),'w')

    print("#name,region,country,lat,lon,location_type,conflict_date,population", file=loc_file)

    for x in range(0,lx):
        for y in range(0,ly):
            name = "{}_{}".format(x,y)
            region = "{}".format(x+y)
            country = "{}".format(x-y)
            lat = x
            lon = y
            location_type = "town"
            if x == 0 or x == lx-1:
                location_type = "camp"
            if y == 0 or y == ly-1:
                location_type = "camp"
            if x > 3.9*lx/10.0 and x < 6.1*lx/10.0:
                location_type = "conflict_zone"
            if y > 3.9*ly/10.0 and y < 6.1*ly/10.0:
                location_type = "conflict_zone"
            conflict_date = 0
            population = 10000000
            print("{},{},{},{},{},{},{},{}".format(name,region,country,lat,lon,location_type,conflict_date,population), file=loc_file)


    routes_file = open("{}routes.csv".format(output_prefix),'w')
    print("#name1,name2,distance,forced_redirection", file=routes_file)

    for x in range(0,lx):
        for y in range(0,ly):
            name1 = "{}_{}".format(x,y)
            forced_redirection = 0
            if degree>1 and x+1 < lx:
                name2 = "{}_{}".format(x+1,y)
                distance = random.randint(50,200)
                print("{},{},{},{}".format(name1, name2, distance, forced_redirection), file=routes_file)
            if degree>3 and y+1 < ly:
                name2 = "{}_{}".format(x,y+1)
                distance = random.randint(50,200)
                print("{},{},{},{}".format(name1, name2, distance, forced_redirection), file=routes_file)
            if degree>5 and x+1 < lx and y+1 < ly:
                name2 = "{}_{}".format(x+1,y+1)
                distance = random.randint(50,200)
                print("{},{},{},{}".format(name1, name2, distance, forced_redirection), file=routes_file)
            if degree>7 and x+1 < lx and y-1 > 0:
                name2 = "{}_{}".format(x+1,y-1)
                distance = random.randint(50,200)
                print("{},{},{},{}".format(name1, name2, distance, forced_redirection), file=routes_file)



if __name__ == '__main__':
    prefix = ""
    if len(sys.argv) > 4:
        prefix = sys.argv[4]
    generate_grid_graph(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), prefix)
