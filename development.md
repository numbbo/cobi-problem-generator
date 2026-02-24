## How to version the project and create a GitHub release

### 1. Update the version number in project files

In `pyproject.toml`:
```
version = "1.0.0"
```

In `cobi/__init__.py`:
```
__version__ = "1.0.0"
```

### 2. Create a version (tag) on git

```
git add .
git commit -m "Release version 1.0.0"
git tag v1.0.0
git push origin v1.0.0
```

### 3. Create a GitHub release

On GitHub:
1. Go to your repo
2. Click Releases
3. Click Draft a new release
4. Select tag v1.0.0
5. Add release notes
6. Publish

This creates a proper versioned release page.