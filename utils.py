# imports


# file name from path
def get_file_name_from_path(path: str, extension: bool = False):
  filename = path.split('/')[-1]
  filename = filename if extension else filename.split('.')[0]
  return filename


# run file
if __name__ == '__main__':
  pass
