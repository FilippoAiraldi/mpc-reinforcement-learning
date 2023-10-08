find . -type f -name "*.py" | while read -r file; do
    pyupgrade --py39-plus "$file"
done
autoflake --remove-all-unused-imports --remove-unused-variables --remove-duplicate-keys --in-place --recursive .
isort .
black .
coverage run -m unittest discover tests
coverage xml
