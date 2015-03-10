#! /usr/bin/env python


import os


def _kwargs(kwargs, key, value):
    if key not in kwargs.keys():
        kwargs[key] = value


def _parseFilename(path):
    name, ext = os.path.splitext(path)
    ext = ext[1:].lower()
    return name, ext


class Directory(object):

    def __init__(self, directory_path, **kwargs):
        _kwargs(kwargs, "include_file_extensions", None)
        _kwargs(kwargs, "include_hidden", False)
        _kwargs(kwargs, "exclude_file_extensions", [])
        _kwargs(kwargs, "recursive", False)

        if(kwargs['include_file_extensions'] == 'images'):
            kwargs['include_file_extensions'] = ['jpg', 'jpeg', 'png', 'tiff']

        if not os.path.exists(directory_path):
            raise Exception("DIRECTORY_EXISTENCE", "The directory path you specified [" + directory_path + "] does not exist.")

        self.directory_path = directory_path
        self.absolute_directory_path = os.path.abspath(directory_path)
        self.recursive = kwargs["recursive"]

        self.file_list = []
        self.directory_list = []
        for line in os.listdir(self.absolute_directory_path):
            line_path = os.path.join(self.absolute_directory_path, line)
            filename, line_extension = _parseFilename(line_path)

            if os.path.isfile(line_path) and \
               (kwargs["include_hidden"] or line[0] != ".") and \
               (kwargs["include_file_extensions"] is None or line_extension in kwargs["include_file_extensions"]) and \
               (line_extension not in kwargs["exclude_file_extensions"]):
                    self.file_list.append(line_path)
            elif os.path.isdir(line_path) and \
                 kwargs["recursive"] and \
                 (kwargs["include_hidden"] or line[0] != "."):
                    self.directory_list.append(Directory(line_path, **kwargs))

    def __str__(self):
        return "<Directory Object | %s | %d files | %d directories>" % (self.absolute_directory_path, len(self.file_list), len(self.directory_list))

    def __repr__(self):
        return "<Directory Object | ../%s>" % (self.base())

    def __iter__(self):
        for filename in self.files():
            yield filename

    def base(self):
        return os.path.basename(self.absolute_directory_path)

    def files(self, **kwargs):
        _kwargs(kwargs, "recursive", self.recursive)

        directory_files = []
        if kwargs["recursive"]:
            for directory in self.directory_list:
                directory_files += directory.files(**kwargs)

        return self.file_list + directory_files

    def directories(self, **kwargs):
        _kwargs(kwargs, "recursive", self.recursive)

        directory_dirs = []
        if kwargs["recursive"]:
            for directory in self.directory_list:
                directory_dirs += directory.directories(**kwargs)

        return self.directory_list + directory_dirs

    def num_files(self, **kwargs):
        _kwargs(kwargs, "recursive", self.recursive)

        directories_num_files = 0

        if kwargs["recursive"]:
            for directory in self.directory_list:
                directories_num_files += directory.num_files(**kwargs)

        return len(self.file_list) + directories_num_files

    def num_directories(self, **kwargs):
        _kwargs(kwargs, "recursive", self.recursive)

        directories_num_directories = 0

        if kwargs["recursive"]:
            for directory in self.directory_list:
                directories_num_directories += directory.num_directories(**kwargs)

        return len(self.directory_list) + directories_num_directories


if __name__ == "__main__":
    # Test directory does not exist
    try:
        direct = Directory("test/")
    except Exception as e:
        assert e[0] == "DIRECTORY_EXISTENCE"

    # Test recursive
    direct = Directory("tests/directory", recursive=True)
    assert direct.num_files(recursive=False) == 10
    assert direct.num_files() == 15
    assert direct.num_directories(recursive=False) == 2
    assert direct.num_directories() == 5

    # Test not recursive
    direct = Directory("tests/directory")
    assert direct.num_files(recursive=False) == 10
    assert direct.num_files() == 10
    assert direct.num_directories(recursive=False) == 0
    assert direct.num_directories() == 0

    # Test include
    direct = Directory("tests/directory", include_file_extensions=["txt"], recursive=True)
    assert direct.num_files(recursive=False) == 3
    assert direct.num_files() == 6

    # Test exclude
    direct = Directory("tests/directory", exclude_file_extensions=["ignore"], recursive=True)
    assert direct.num_files(recursive=False) == 9
    assert direct.num_files(recursive=True) == 13
