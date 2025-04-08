

def beta_strand_determination(pdb_file, output_file):

  with open(pdb_file, 'r') as file:
    lines = file.readlines()


  strands = ["B", "C", "E", "F"]
  strand_id = 0
  prev_resID = None
  updated_lines = []

  lineCount = 0
  for line in lines:
    if line.startswith("ATOM"):
      lineCount += 1
      resID = int(line[22:26].strip())


      if lineCount > 1 and resID > prev_resID + 1:
        strand_id = (strand_id + 1) % len(strands)
      elif lineCount < 2:
        prev_resID = resID


      new_line = line[:21] + strands[strand_id] + line[22:]
      updated_lines.append(new_line)
      prev_resID = resID



  with open(output_file, 'w') as file:
    file.writelines(updated_lines)



#main code starts here
if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser(description='Conserved beta strand determination')
  parser.add_argument("-i",required = True)
  parser.add_argument("-o", required = True)

  args = parser.parse_args()
  beta_strand_determination(args.i, args.o)
