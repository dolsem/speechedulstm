import wave, struct, csv, time
import pickle

root_path = "./db/SantaBarbara/"
#this parses the transcript into a list of dictionaries for each time slice
def timeParse(i):
    with open(root_path+"TRN/SBC%03d.trn" % (i)) as f:
        name = ""
        result = []
        for line in f.readlines():
            try:
                time, n, words = line.split("\t")
            except ValueError:
                print "Value Error: %s" % (line.strip())
                continue
            start, end = map(float, time.split())
            name = n.strip().strip(":") if not n.isspace() else name
            result+=[{"start": start, "end": end, "name": name.upper(), "words": words.rstrip("\n")}]
        return result

#this parses the metadata files with the given headers and returns them as a dictionary
#name maps to row data as dictionary
def metaParse(i):
    result = {}
    dupes = set()
    with open(root_path+"meta/metadata%d.csv" % i, 'rU') as f:
        for row in csv.DictReader(f):
            if row.get('name').upper() in result.keys() or row.get('name').upper() in dupes:
                result.pop(row.get('name').upper())
                dupes.update(row.get('name').upper())
            else:
                result.update({row.get('name').upper() : row })
    return result

#this parses the audio .wav files with the "wave" library and the "struct" library to unpack the byte stream
#pulls all samples from in the selection size, storing it as a list of tuples from the channels
def audioParseSelection(i, start, end):
    w = wave.open(root_path+"wav/SBC%03d.wav" % (i), 'r')
    w.setpos(int(start * w.getframerate()))
    try:
        raw = [struct.unpack("%dh" % (w.getnchannels()), w.readframes(1)) for _ in range(int(w.getframerate() * (end - start)))]
        w.close()
        return {"left": [x[0] for x in raw], "right": [x[0] for x in raw]}, w.getframerate()
    except Exception as e:
        print "%s at %d from %.2f to %.2f" % (str(e), i, start, end)
        return [],0

#this combines all three data sources together into individual dictionary elements
#based on the times slices, it then pulls in all available metadata (matching the names)
#and audio values from the wave (using the start and end seconds)
def combine (i, time, meta):
    #print "%d, %s: %f to %f" % (i, time.get("name"), time.get("start"), time.get("end"))
    audio, freq = audioParseSelection(i, time.get("start"), time.get("end"))
    if time.get("name") in meta.keys() and len(audio) > 0:
        time.update(meta.get(time.get("name")))
        time.update({"wav" : audio, "nframes" : len(audio)})
        time.update({"scene_id" : i, "freq": freq})
        return time
    else:
        print "data not found:", time.get("name"), time.get("start"), time.get("end"), len(audio)

def pickleData(meta, r):
    for i in r:
        f = open(root_path + "sb%d.p" % (i), "wb")
        times = timeParse(i)
        print "i: %d, total: %d" % (i, len(times))
        pickle.dump([combine(i, dict(time), meta) for time in times], f)
        f.close()

ranges = [range(1,15), range(15,31), range(31, 47), range(47, 61)]

for x in [1, 2, 3, 4]:
    print("Processing metadata #{}, conversations {}...".format(x, ranges[x-1]))
    pickleData(metaParse(x), ranges[x-1])
    print "Done.\n"
