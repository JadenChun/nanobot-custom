# Folder access and workspaces

Nanobot keeps instance data, agent files, and chat project folders as separate concepts. Knowing which folder serves each purpose helps you choose a narrow project folder when a task requests access.

## Where Nanobot keeps data

- **Instance data directory** stores configuration, sessions, logs, and WebUI folder-grant state. In the current example installation this is `/home/jaden/.nanobot-staging`.
- **Agent home** stores the agent's profiles, memory, skills, and default output files. In this installation it is `/home/jaden/.nanobot-staging/workspace`.
- **Per-chat project folder** is the working folder selected for one chat. It can be outside Agent home. Selecting it changes that chat's working folder; it does not grant filesystem access or move session memory.

The paths above describe the current example layout. They do not indicate that Nanobot migrated or reorganized any files.

Agent home is shown in **Settings → Runtime**, including guidance for changing `agents.defaults.workspace` in the configuration file and restarting Nanobot. Changing this setting does not move existing agent files automatically.

The chat composer keeps Agent home out of everyday project controls. With no project selected, use the folder button to choose a project folder. A selected project is shown as **Working in** and can be cleared without changing the Agent home configuration.

## Choose a folder for a task

When a task identifies a blocked folder or file, review the exact path and purpose shown in its access request. For a file request, Nanobot shows both the original file and the containing folder that an approval would cover. If the target is unknown, choose a specific project folder yourself. A project selection is not an access grant.

Grant only the capabilities the task needs. Read access lets Nanobot inspect files, write access also lets it modify files, and command access allows commands to run against the approved folder. Write tasks need write permission, while command tasks need command permission.

## One-time and saved access

- **Allow for this task** creates a task-bound Once permission. It applies to the approved folder and capabilities for the current task, then expires with that task's permission lifecycle.
- **Always allow** saves a reusable folder grant in this Nanobot instance. The grant uses the file and command permissions you reviewed and can be reused by other chats in the same instance.

Choose a project folder rather than a broad home, root, or runtime-data folder. Nanobot keeps private runtime storage outside the folder grant boundary.
