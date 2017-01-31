#!/usr/bin/env bash

module="safe_learning"

get_script_dir () {
     SOURCE="${BASH_SOURCE[0]}"
     # While $SOURCE is a symlink, resolve it
     while [ -h "$SOURCE" ]; do
          DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
          SOURCE="$( readlink "$SOURCE" )"
          # If $SOURCE was a relative symlink (so no "/" as prefix, need to resolve it relative to the symlink base directory
          [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE"
     done
     DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
     echo "$DIR"
}

# Change to script root
cd $(get_script_dir)
GREEN='\033[0;32m'
NC='\033[0m'

# Run style tests
echo -e "${GREEN}Running style tests.${NC}"
flake8 $module --exclude test*.py,__init__.py --ignore=E402,W503 --show-source

# Ignore import errors for __init__ and tests
flake8 $module --filename=__init__.py,test*.py --ignore=F,E402,W503 --show-source

echo -e "${GREEN}Testing docstring conventions.${NC}"
# Test docstring conventions
pydocstyle $module --match='(?!__init__).*\.py' 2>&1 | grep -v "WARNING: __all__"

# Run unit tests
echo -e "${GREEN}Running unit tests.${NC}"
nosetests --with-doctest --nocapture --with-coverage --cover-min-percentage=80 --cover-erase --cover-package=$module $module

# Export html
coverage html

