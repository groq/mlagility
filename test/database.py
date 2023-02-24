import os
import shutil
import unittest
from pathlib import Path
import yaml
from groqflow.common import cache
import mlagility.common.filesystem as fs

# Create a test directory and make it the CWD
test_dir = "database_test_dir"
cache_dir = "cache-dir"
dirpath = Path(test_dir)
if dirpath.is_dir():
    shutil.rmtree(dirpath)
os.makedirs(test_dir)
os.chdir(test_dir)

corpus_dir = os.path.join(os.getcwd(), "test_corpus")
extras_dir = os.path.join(corpus_dir, "extras")
os.makedirs(extras_dir, exist_ok=True)

database_file = ".cache_database.yaml"


class Testing(unittest.TestCase):
    def setUp(self) -> None:
        cache.rmdir(cache_dir)
        fs.make_cache_dir(cache_dir)
        return super().setUp()

    def test_001_add_script(self):
        script_name = "test_script"

        db = fs.CacheDatabase(cache_dir)
        db.add_script(script_name)

        database_path = os.path.join(cache_dir, database_file)

        with open(database_path, "r", encoding="utf8") as stream:
            database = yaml.load(stream, Loader=yaml.FullLoader)

        assert script_name in database.keys()

    def test_002_add_build(self):
        script_name = "test_script"
        build_name = "test_build"

        db = fs.CacheDatabase(cache_dir)
        db.add_build(script_name, build_name)

        database_path = os.path.join(cache_dir, database_file)

        with open(database_path, "r", encoding="utf8") as stream:
            database = yaml.load(stream, Loader=yaml.FullLoader)

        assert script_name in database.keys()
        assert build_name in database[script_name].keys()

    def test_003_add_multiple_scripts(self):
        script_names = ["test_script_1", "test_script_2", "test_script_3"]

        db = fs.CacheDatabase(cache_dir)

        for script_name in script_names:
            db.add_script(script_name)

        database_path = os.path.join(cache_dir, database_file)

        with open(database_path, "r", encoding="utf8") as stream:
            database = yaml.load(stream, Loader=yaml.FullLoader)

        for script_name in script_names:
            assert script_name in database.keys()

    def test_004_add_multiple_builds(self):
        script_name = "test_script"
        build_names = ["test_build_1", "test_build_2", "test_build_3"]

        db = fs.CacheDatabase(cache_dir)

        for build_name in build_names:
            db.add_build(script_name, build_name)

        database_path = os.path.join(cache_dir, database_file)

        with open(database_path, "r", encoding="utf8") as stream:
            database = yaml.load(stream, Loader=yaml.FullLoader)

        assert script_name in database.keys()

        for build_name in build_names:
            assert build_name in database[script_name].keys()

    def test_005_add_multiple_builds_and_scripts(self):
        script_names = ["test_script_1", "test_script_2"]
        build_names = {
            script_names[0]: ["test_build_1", "test_build_2", "test_build_3"],
            script_names[1]: ["test_build_4", "test_build_5", "test_build_6"],
        }

        db = fs.CacheDatabase(cache_dir)

        for script_name in script_names:
            for build_name in build_names[script_name]:
                db.add_build(script_name, build_name)

        database_path = os.path.join(cache_dir, database_file)

        with open(database_path, "r", encoding="utf8") as stream:
            database = yaml.load(stream, Loader=yaml.FullLoader)

        for script_name in script_names:
            assert script_name in database.keys()
            for build_name in build_names[script_name]:
                assert build_name in database[script_name].keys()

    def test_006_remove_build(self):
        script_name = "test_script"
        build_name = "test_build"

        db = fs.CacheDatabase(cache_dir)
        db.add_build(script_name, build_name)

        database_path = os.path.join(cache_dir, database_file)

        # Make sure the build is there in the first place
        with open(database_path, "r", encoding="utf8") as stream:
            database = yaml.load(stream, Loader=yaml.FullLoader)

        assert script_name in database.keys()
        assert build_name in database[script_name].keys()

        # Remove the build and then make sure it's gone and
        # the script is gone too
        db.remove_build(build_name)

        with open(database_path, "r", encoding="utf8") as stream:
            database = yaml.load(stream, Loader=yaml.FullLoader)

        assert script_name not in database.keys()
        assert len(database) == 0

    def test_007_remove_one_build_from_multiple_builds(self):
        script_name = "test_script"
        build_names = ["test_build_1", "test_build_2", "test_build_3"]

        db = fs.CacheDatabase(cache_dir)

        for build_name in build_names:
            db.add_build(script_name, build_name)

        database_path = os.path.join(cache_dir, database_file)

        # Make sure all builds are there in the first place
        with open(database_path, "r", encoding="utf8") as stream:
            database = yaml.load(stream, Loader=yaml.FullLoader)

        assert script_name in database.keys()

        for build_name in build_names:
            assert build_name in database[script_name].keys()

        # Remove one build. Make sure that build is gone, but the
        # script and other builds are still there
        build_to_remove = build_names[0]
        db.remove_build(build_to_remove)

        with open(database_path, "r", encoding="utf8") as stream:
            database = yaml.load(stream, Loader=yaml.FullLoader)

        assert script_name in database.keys()
        for build_name in build_names:
            if build_name != build_to_remove:
                assert build_name in database[script_name].keys()

    def test_008_remove_one_script_from_multiple_scripts(self):
        script_names = ["test_script_1", "test_script_2"]
        build_names = {
            script_names[0]: ["test_build_1", "test_build_2", "test_build_3"],
            script_names[1]: ["test_build_4", "test_build_5", "test_build_6"],
        }

        db = fs.CacheDatabase(cache_dir)

        for script_name in script_names:
            for build_name in build_names[script_name]:
                db.add_build(script_name, build_name)

        database_path = os.path.join(cache_dir, database_file)

        # Make sure the builds and scripts are all there
        with open(database_path, "r", encoding="utf8") as stream:
            database = yaml.load(stream, Loader=yaml.FullLoader)

        for script_name in script_names:
            assert script_name in database.keys()
            for build_name in build_names[script_name]:
                assert build_name in database[script_name].keys()

        # Remove one script. Make sure that script is gone, but the other
        # script and its builds are still there
        script_to_remove = script_names[0]
        for build_to_remove in build_names[script_to_remove]:
            db.remove_build(build_to_remove)

        with open(database_path, "r", encoding="utf8") as stream:
            database = yaml.load(stream, Loader=yaml.FullLoader)

        for script_name in script_names:
            if script_name != script_to_remove:
                assert script_name in database.keys()
                for build_name in build_names[script_name]:
                    assert build_name in database[script_name].keys()

    def test_009_add_build_after_remove(self):
        script_name = "test_script"
        first_build_name = "test_build"
        second_build_name = "test_build_2"

        db = fs.CacheDatabase(cache_dir)
        db.add_build(script_name, first_build_name)

        database_path = os.path.join(cache_dir, database_file)

        # Make sure the first build is there in the first place
        with open(database_path, "r", encoding="utf8") as stream:
            database = yaml.load(stream, Loader=yaml.FullLoader)

        assert script_name in database.keys()
        assert first_build_name in database[script_name].keys()

        # Remove the first build and then make sure it's gone and
        # the script is gone too
        db.remove_build(first_build_name)

        with open(database_path, "r", encoding="utf8") as stream:
            database = yaml.load(stream, Loader=yaml.FullLoader)

        assert script_name not in database.keys()
        assert len(database) == 0

        # Add the second build and make sure it's there
        db.add_build(script_name, second_build_name)

        with open(database_path, "r", encoding="utf8") as stream:
            database = yaml.load(stream, Loader=yaml.FullLoader)

        assert script_name in database.keys()
        assert second_build_name in database[script_name].keys()

    def test_010_script_in_database(self):
        script_name = "test_script"

        db = fs.CacheDatabase(cache_dir)
        db.add_script(script_name)

        assert db.script_in_database(script_name)

    def test_011_database_exists(self):
        script_name = "test_script"

        db = fs.CacheDatabase(cache_dir)

        # Database should not exist on disk before we put any contents in it
        assert not db.exists()

        db.add_script(script_name)

        # Now the database should exist
        assert db.exists()


if __name__ == "__main__":
    unittest.main()
