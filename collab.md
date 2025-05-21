# Collaboration Workflow Guide
## 1. Initial Setup
Before you start, make sure both collaborators have cloned the repository from GitHub.
### Clone the repo:
```bash
git clone https://github.com/your-username/project-name.git
```

## 2. Daily Workflow
Before you start working, always ensure your local version is up-to-date with the main branch. This prevents merge conflicts.
### With Git Extensions GUI:
Open Git Extensions and ensure your working directory is the project folder.
Pull the latest version from the main branch:
Click on "Pull" from the toolbar.
Make sure you're pulling from the main branch.
### With Command Line:
Open the terminal and navigate to your project folder.
Run the following to ensure you're up-to-date with the main branch:
```bash
git checkout main
git pull origin main
```

## 3. Creating a Branch for Your Work
Each time you start working on a new task or feature, create a new branch from main. This ensures that you won't directly touch main and can work freely without affecting each other.
### With Git Extensions GUI:
Open Git Extensions and click on "Checkout" (in the toolbar).
Choose “Create New Branch,” give it a meaningful name, and switch to it.
### With Command Line:
Run this command to create and switch to a new branch:
```bash
git checkout -b branch_name
```

## 4. Work on Your Feature
Now, you’re ready to work on your code. All collaborators will work on their respective tasks in their own branches.
**Commit Frequently:**
Make commits often, ideally after completing small tasks or units of work. This helps track progress and makes it easier to resolve conflicts later. Add meaningful comments to your commits.
### Commit examples with Command Line:
```bash
git commit -m "Added feature XY"
git commit -m "fixed bug where XY happened"
git commit -m "restructured code to be more readable"
```
### Commit with Git Extensions GUI:
Make sure you are in the correct branch.
Click on "Commit".
Stage the relevant changes.
Enter a commit message.
Click "Commit"

## 5. Pushing Your Work
Once you’re done or have made meaningful progress, push your branch to GitHub.

### With Git Extensions GUI:
Click on "Push" in Git Extensions.
Make sure your branch is selected, then push it to GitHub.
### With Command Line:
Push your branch to GitHub:
```bash
git push origin branch_name
```

## 6. Merging Changes to main
Once your feature is complete (or at a milestone), it’s time to merge your work into main.
**Open a Pull Request:**
Go to GitHub → open the repository → click on "Pull Requests" → "New Pull Request" → select your branch → submit the Pull Request to main.
**Review & Merge:**
One collaborator reviews the changes (if necessary, you can do this together).
Once reviewed, click "Merge" to merge your feature branch into main.

## 7. Deleting Your Branch
After the pull request is merged, delete your feature branch both locally and remotely to keep things tidy.
### With Git Extensions GUI:
Go to the "Branches" tab → right-click on the feature branch → click "Delete."
### With Command Line:
To delete the branch locally:
```bash
git branch -d feature/ml-integration
```
To delete the branch remotely:
```bash
git push origin --delete feature/ml-integration
```

## 8. Stay Up-to-Date
Before you start working again, repeat step 2: always ensure you're pulling the latest version of main to stay in sync with your collaborator’s changes.

## Additional Tips for Better Collaboration:
**Commit messages:** Always write clear, concise commit messages. Focus on what changed and why.
**Rebase (optional for advanced users):** If you’re working in parallel on features that might conflict, rebasing (git pull --rebase) can help keep things cleaner than merging.
**Communication:** Let each other know when you’re starting and finishing tasks. _Use GitHub issues_ or a shared document to track tasks.
**Main branch:** The main branch is only for the _stable_ releases, that can be used. Only merge branches if the new feature is ready to use or feature-changes have been tested or approved to work.
**Bug reports:** Use GitHub issues to report any errors.