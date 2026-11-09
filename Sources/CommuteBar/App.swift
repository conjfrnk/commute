import AppKit
import Combine
import SwiftUI

@main
struct CommuteBarApp {
    @MainActor static func main() {
        // Settings is opened explicitly, never as an initial or restored SwiftUI scene.
        UserDefaults.standard.set(false, forKey: "NSQuitAlwaysKeepsWindows")
        let app = NSApplication.shared
        app.setActivationPolicy(.accessory)
        let delegate = AppDelegate()
        app.delegate = delegate
        withExtendedLifetime(delegate) { app.run() }
    }
}
@MainActor
final class CommutePanel: NSPanel {
    override var canBecomeKey: Bool { true }
    override var canBecomeMain: Bool { false }
}
@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    private var statusItem: NSStatusItem!
    private var popup: NSPanel!
    private var outsideClickMonitor: Any?
    private var localEventMonitor: Any?
    private let model = CommuteModel()
    private var subscription: AnyCancellable?
    private var terminationSignal: DispatchSourceSignal?
    private var settingsWindow: NSWindow?
    func applicationDidFinishLaunching(_ notification: Notification) {
        if CommandLine.arguments.contains("--check-data") || CommandLine.arguments.contains("--render-preview")
            || CommandLine.arguments.contains("--render-settings")
        {
            do {
                let p = Process()
                let pipe = Pipe()
                p.executableURL = model.root.appendingPathComponent("venv/bin/python")
                p.arguments = [model.root.appendingPathComponent("commute_data.py").path]
                p.standardOutput = pipe
                try p.run()
                let data = pipe.fileHandleForReading.readDataToEndOfFile()
                p.waitUntilExit()
                let decoder = JSONDecoder()
                decoder.keyDecodingStrategy = .convertFromSnakeCase
                let snapshot = try decoder.decode(Snapshot.self, from: data)
                guard p.terminationStatus == 0, snapshot.routes["to_work"] != nil, snapshot.routes["to_home"] != nil else {
                    throw NSError(domain: "Smoke", code: 1)
                }
                model.snapshot = snapshot
                let savedSelection = model.selection  // setting persists; restored below so a render cannot change it
                model.selection = "auto"  // a persisted manual direction must not leak into the render
                model.detectedDirection = CommandLine.arguments.contains("--preview-to-work") ? "to_work" : "to_home"
                model.locationText = model.detectedDirection == "to_work" ? "At home · heading to work" : "At work · heading home"
                model.samplerText = "Sampling every 10 min"
                let settings = try CommuteSettings.read(root: model.root)
                let encoder = JSONEncoder()
                encoder.keyEncodingStrategy = .convertToSnakeCase
                let settingsData = try encoder.encode(settings)
                let decodedSettings = try decoder.decode(CommuteSettings.self, from: settingsData)
                guard decodedSettings.home == settings.home, decodedSettings.toWorkWaypoints == settings.toWorkWaypoints else {
                    throw NSError(domain: "Settings", code: 1)
                }
                if CommandLine.arguments.contains("--render-settings") {
                    let view = NSHostingView(rootView: SettingsView(model: model, preloaded: settings))
                    view.frame = NSRect(origin: .zero, size: view.fittingSize)
                    view.layoutSubtreeIfNeeded()
                    guard let bitmap = view.bitmapImageRepForCachingDisplay(in: view.bounds) else {
                        throw NSError(domain: "Render", code: 1)
                    }
                    view.cacheDisplay(in: view.bounds, to: bitmap)
                    try bitmap.representation(using: .png, properties: [:])!.write(
                        to: model.root.appendingPathComponent("build/settings-preview.png"))
                }
                if CommandLine.arguments.contains("--render-preview") {
                    let view = NSHostingView(
                        rootView: CommutePopover(
                            model: model, previewHover: CommandLine.arguments.contains("--preview-hover") ? 1020 : nil,
                            previewMode: CommandLine.arguments.contains("--preview-today")
                                ? "today" : CommandLine.arguments.contains("--preview-model") ? "model" : nil))
                    view.frame = NSRect(origin: .zero, size: view.fittingSize)
                    view.layoutSubtreeIfNeeded()
                    guard let bitmap = view.bitmapImageRepForCachingDisplay(in: view.bounds) else {
                        throw NSError(domain: "Render", code: 1)
                    }
                    view.cacheDisplay(in: view.bounds, to: bitmap)
                    try bitmap.representation(using: .png, properties: [:])!.write(
                        to: model.root.appendingPathComponent("build/preview.png"))
                }
                model.selection = savedSelection
                let models = snapshot.routes.sorted { $0.key < $1.key }.map { "\($0.key)=\($0.value.modelLabel) [\($0.value.modelState)]" }
                    .joined(separator: ", ")
                let curvePoints = snapshot.routes.values.reduce(0) { $0 + $1.history.count }  // the points the Predicted graph draws
                print(
                    "Snapshot decoded: both routes, \(snapshot.routes.values.reduce(0) { $0 + $1.today.count }) observations today, \(curvePoints) curve points; day status \(snapshot.dayStatus ?? "n/a"); model \(models); fractional date parsing: \(parseDate(snapshot.generatedAt) != nil)"
                )
                exit(0)
            } catch {
                fputs("Smoke check failed: \(error)\n", stderr)
                exit(1)
            }
        }
        installMainMenu()
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        if let button = statusItem.button {
            button.image = NSImage(systemSymbolName: "car.fill", accessibilityDescription: "Commute")
            button.imagePosition = .imageLeading
            button.target = self
            button.action = #selector(toggle)
            button.setAccessibilityLabel("Commute estimates")
        }
        let controller = NSHostingController(rootView: CommutePopover(model: model, onSettings: { [weak self] in self?.showSettings() }))
        popup = CommutePanel(contentRect: .zero, styleMask: [.borderless, .nonactivatingPanel], backing: .buffered, defer: false)
        popup.contentViewController = controller
        popup.backgroundColor = .clear
        popup.isOpaque = false  // Transparent rounded corners; the SwiftUI surface itself is solid.
        popup.hasShadow = true
        popup.level = .popUpMenu
        popup.isReleasedWhenClosed = false
        popup.collectionBehavior = [.transient, .moveToActiveSpace, .fullScreenAuxiliary]
        outsideClickMonitor = NSEvent.addGlobalMonitorForEvents(matching: [.leftMouseDown, .rightMouseDown]) { [weak self] _ in
            Task { @MainActor in
                guard let self else { return }
                // A click on the status item is toggle()'s to handle: closing here on mouse-down would make its
                // action reopen the popup. The item's window frame is used, not the button's bounds: clicks in
                // the padding around the icon still reach the button's action.
                if let rect = self.statusItemScreenRect(), rect.contains(NSEvent.mouseLocation) { return }
                self.popup.orderOut(nil)
            }
        }
        localEventMonitor = NSEvent.addLocalMonitorForEvents(matching: [.leftMouseDown, .rightMouseDown, .keyDown]) { [weak self] event in
            guard let self, self.popup.isVisible else { return event }
            if event.type == .keyDown && event.keyCode == 53 {
                self.popup.orderOut(nil)
                return nil
            }
            if event.type != .keyDown && event.window != self.popup && event.window != self.statusItem.button?.window {
                self.popup.orderOut(nil)
            }
            return event
        }
        subscription = model.objectWillChange.sink { [weak self] in
            DispatchQueue.main.async {
                guard let self else { return }
                self.updateStatusTitle()
                if self.popup.isVisible { self.positionPopup() }
            }
        }
        updateStatusTitle()
        signal(SIGTERM, SIG_IGN)
        let source = DispatchSource.makeSignalSource(signal: SIGTERM, queue: .main)
        source.setEventHandler { NSApp.terminate(nil) }
        source.resume()
        terminationSignal = source
        model.start()
        if model.needsSetup { showSettings() }
    }
    // Launching or reopening the menu-bar app must not create a window.
    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows flag: Bool) -> Bool { false }
    func applicationShouldSaveSecureApplicationState(_ app: NSApplication) -> Bool { false }
    func applicationShouldRestoreSecureApplicationState(_ app: NSApplication) -> Bool { false }
    private func installMainMenu() {
        let menu = NSMenu()
        let appItem = NSMenuItem()
        let appMenu = NSMenu(title: "Commute Bar")
        appItem.submenu = appMenu
        menu.addItem(appItem)
        let settings = NSMenuItem(title: "Settings…", action: #selector(showSettings), keyEquivalent: ",")
        settings.target = self
        appMenu.addItem(settings)
        appMenu.addItem(.separator())
        appMenu.addItem(withTitle: "Hide Commute Bar", action: #selector(NSApplication.hide(_:)), keyEquivalent: "h")
        let hideOthers = appMenu.addItem(
            withTitle: "Hide Others", action: #selector(NSApplication.hideOtherApplications(_:)), keyEquivalent: "h")
        hideOthers.keyEquivalentModifierMask = [.command, .option]
        appMenu.addItem(withTitle: "Show All", action: #selector(NSApplication.unhideAllApplications(_:)), keyEquivalent: "")
        appMenu.addItem(.separator())
        appMenu.addItem(withTitle: "Quit Commute Bar", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")

        let fileItem = NSMenuItem()
        fileItem.submenu = NSMenu(title: "File")
        fileItem.submenu?.addItem(withTitle: "Close Window", action: #selector(NSWindow.performClose(_:)), keyEquivalent: "w")
        menu.addItem(fileItem)

        let editItem = NSMenuItem()
        let editMenu = NSMenu(title: "Edit")
        editItem.submenu = editMenu
        menu.addItem(editItem)
        editMenu.addItem(withTitle: "Undo", action: Selector(("undo:")), keyEquivalent: "z")
        let redo = editMenu.addItem(withTitle: "Redo", action: Selector(("redo:")), keyEquivalent: "z")
        redo.keyEquivalentModifierMask = [.command, .shift]
        editMenu.addItem(.separator())
        editMenu.addItem(withTitle: "Cut", action: #selector(NSText.cut(_:)), keyEquivalent: "x")
        editMenu.addItem(withTitle: "Copy", action: #selector(NSText.copy(_:)), keyEquivalent: "c")
        editMenu.addItem(withTitle: "Paste", action: #selector(NSText.paste(_:)), keyEquivalent: "v")
        editMenu.addItem(withTitle: "Select All", action: #selector(NSText.selectAll(_:)), keyEquivalent: "a")
        NSApp.mainMenu = menu
    }
    @objc private func showSettings() {
        popup.orderOut(nil)
        if settingsWindow == nil {
            let controller = NSHostingController(rootView: SettingsView(model: model))
            let window = NSWindow(contentViewController: controller)
            window.title = "Commute settings"
            window.styleMask = [.titled, .closable]
            window.isReleasedWhenClosed = false
            window.isRestorable = false
            // Size the window to the content now: the default frame is too small, and a window that resizes
            // after centering drifts down and right since its top-left corner stays put.
            window.setContentSize(
                controller.sizeThatFits(in: CGSize(width: CGFloat.greatestFiniteMagnitude, height: CGFloat.greatestFiniteMagnitude)))
            settingsWindow = window
        }
        // Center on the screen the user is looking at (the one holding the status item they clicked), on every
        // open, not just the first: the window survives being closed, so wherever it was left it would reopen.
        if let window = settingsWindow {
            window.makeKeyAndOrderFront(nil)  // ordering front is what gives the content its real size
            centerSettings(window)
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.15) { [weak self, weak window] in
                if let window { self?.centerSettings(window) }  // catch any resize committed after the layout pass
            }
        }
        NSApp.activate(ignoringOtherApps: true)
    }
    /// True center of the screen's visible area (clear of the menu bar and Dock), in the window's current size.
    private func centerSettings(_ window: NSWindow) {
        guard let screen = statusItem.button?.window?.screen ?? window.screen ?? NSScreen.main ?? NSScreen.screens.first else { return }
        let visible = screen.visibleFrame, size = window.frame.size
        window.setFrameOrigin(NSPoint(x: visible.midX - size.width / 2, y: visible.midY - size.height / 2))
    }
    /// The whole status item's rectangle in screen coordinates (a window's frame already is): the region where
    /// clicks are delivered to the item, including the padding around the visible icon.
    private func statusItemScreenRect() -> NSRect? { statusItem.button?.window?.frame }
    /// The menu-bar text: orange when the live forecast is running hot — a glanceable alarm without
    /// opening the popover. Uses the same threshold as the "running hot" notification.
    private func updateStatusTitle() {
        guard let button = statusItem.button else { return }
        button.attributedTitle = NSAttributedString(
            string: model.statusTitle,
            attributes: model.isRunningHot ? [.foregroundColor: NSColor.systemOrange] : [:])
    }
    /// The menu bar icon's rectangle in screen coordinates.
    private func statusButtonRect() -> NSRect? {
        guard let button = statusItem.button, let window = button.window else { return nil }
        return window.convertToScreen(button.convert(button.bounds, to: nil))
    }
    private func positionPopup() {
        guard let buttonRect = statusButtonRect(), let statusWindow = statusItem.button?.window,
            let content = popup.contentView, let screen = statusWindow.screen ?? NSScreen.main ?? NSScreen.screens.first
        else { return }
        let size = content.fittingSize
        let x = min(max(buttonRect.midX - size.width / 2, screen.visibleFrame.minX + 8), screen.visibleFrame.maxX - size.width - 8)
        let y = max(screen.visibleFrame.minY + 8, buttonRect.minY - size.height - 6)
        popup.setFrame(NSRect(x: x, y: y, width: size.width, height: size.height), display: true)
    }
    @objc private func toggle() {
        if popup.isVisible {
            popup.orderOut(nil)
        } else {
            model.load()
            positionPopup()
            popup.makeKeyAndOrderFront(nil)
        }
    }
    func applicationWillTerminate(_ notification: Notification) {
        if let outsideClickMonitor { NSEvent.removeMonitor(outsideClickMonitor) }
        if let localEventMonitor { NSEvent.removeMonitor(localEventMonitor) }
        model.stop()
    }
}
