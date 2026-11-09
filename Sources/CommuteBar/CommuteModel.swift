import AppKit
import Combine
import CoreLocation
import Foundation
import ServiceManagement
@preconcurrency import UserNotifications

@MainActor
final class CommuteModel: NSObject, ObservableObject, @preconcurrency CLLocationManagerDelegate, UNUserNotificationCenterDelegate {
    @Published var snapshot: Snapshot?
    @Published var error: String?
    @Published var locationText = "Locating…"
    @Published var detectedDirection: String?
    @Published var selection = UserDefaults.standard.string(forKey: "direction") ?? "auto" {
        didSet { UserDefaults.standard.set(selection, forKey: "direction") }
    }
    @Published var samplerText = "Starting sampler…"
    @Published var refreshing = false
    /// Which graph the popover shows ("typical" = the window forecast, "today" = the all-day view). Kept on the
    /// model so a tab switch fires objectWillChange and the panel is re-framed to the new content height.
    @Published var chartMode = "typical"
    /// No commute_settings.json yet: the app shows the welcome/setup branch and spawns no Python.
    @Published var needsSetup = false
    private let locationManager = CLLocationManager()
    private let geocoder = CLGeocoder()
    private var home: CLLocation?
    private var work: CLLocation?
    private var lastLocation: CLLocation?
    private var sampler: Process?
    private var timer: Timer?
    private var wakeObserver: NSObjectProtocol?
    private var activity: NSObjectProtocol?
    private var lastWatchdogAction = Date()  // app launch counts as a fresh start
    private let staleAfter: TimeInterval = 20 * 60  // two sample intervals plus retry slack
    private let reloadInterval: TimeInterval = 60
    private let dataTimeout: TimeInterval = 60  // a normal read takes well under a second
    private var lastLoad = Date.distantPast
    private var lastSignature: Date?
    private var loading = false
    private var dataRevision = 0
    private var reloadPending = false
    private var stopping = false
    private var notifyEnabled = true
    private var lastGeocodeFailure: Date?
    let root: URL
    var newestSample: Date? { snapshot?.routes.values.compactMap { $0.latest?.date }.max() }
    var direction: String? { selection == "auto" ? detectedDirection : selection }
    var route: Route? { direction.flatMap { snapshot?.routes[$0] } }
    var destination: String { direction == "to_home" ? "Home" : "Work" }
    /// True when today's live forecast runs at least the driving allowance over the morning's — the same
    /// threshold the "running hot" notification uses, so the menu bar tint and the alert agree.
    var isRunningHot: Bool {
        guard let route else { return false }
        return (route.forecastOffsetMinutes ?? 0) >= max(4, readNumberSetting("extra_drive_minutes") ?? 5)
    }
    var statusTitle: String {
        guard let direction else { return "Commute" }
        let label = direction == "to_home" ? "Home" : "Work"
        guard let latest = route?.latest else { return "\(label) —" }
        let stale = Date().timeIntervalSince(latest.date) > 15 * 60
        return "\(label) \(Int(latest.minutes))m\(stale ? "*" : "")"
    }
    override init() {
        let path = Bundle.main.object(forInfoDictionaryKey: "CommuteProjectPath") as? String
        root = URL(fileURLWithPath: path ?? FileManager.default.currentDirectoryPath)
        super.init()
        locationManager.delegate = self
        locationManager.desiredAccuracy = kCLLocationAccuracyHundredMeters
        locationManager.distanceFilter = 100  // home/work are matched within 600 m; no need to wake for every small fix
    }
    func start() {
        load()
        startSampler()
        notifyEnabled = readNotifySetting()
        requestNotificationAuthorization()
        if locationManager.authorizationStatus == .notDetermined { locationManager.requestWhenInUseAuthorization() }
        locationManager.startUpdatingLocation()
        // Without this, App Nap can stretch the 15 s health check to minutes while no window is visible.
        activity = ProcessInfo.processInfo.beginActivity(
            options: .userInitiatedAllowingIdleSystemSleep, reason: "Keep commute sampler supervised")
        wakeObserver = NSWorkspace.shared.notificationCenter.addObserver(
            forName: NSWorkspace.didWakeNotification, object: nil, queue: .main
        ) { [weak self] _ in
            Task { @MainActor in
                guard let self, !self.stopping else { return }
                self.restartSamplerAfterWake()
            }
        }
        timer = Timer.scheduledTimer(withTimeInterval: 15, repeats: true) { [weak self] _ in
            Task { @MainActor in
                guard let self, !self.stopping else { return }
                self.startSampler()
                self.checkSamplerHealth()
                self.loadIfNeeded()
            }
        }
    }
    func stop() {
        stopping = true
        timer?.invalidate()
        locationManager.stopUpdatingLocation()
        if let wakeObserver { NSWorkspace.shared.notificationCenter.removeObserver(wakeObserver) }
        if let activity { ProcessInfo.processInfo.endActivity(activity) }
        sampler?.terminate()
    }
    /// A running sampler that stops producing samples (hung Chromium, dead network session) is replaced.
    private func checkSamplerHealth() {
        let now = Date()
        guard now.timeIntervalSince(max(newestSample ?? .distantPast, lastWatchdogAction)) > staleAfter else { return }
        lastWatchdogAction = now
        samplerText = "Sampler stalled · restarting"
        if let sampler, sampler.isRunning {
            sampler.terminate()
        }  // the next tick starts a fresh one
        else {
            stopExternalSampler()
        }
    }
    private var lockPath: String { root.appendingPathComponent(".commute-sampler.lock").path }
    private func lockIsHeld() -> Bool {
        let fd = open(lockPath, O_RDONLY)
        guard fd >= 0 else { return false }
        defer { close(fd) }
        if flock(fd, LOCK_SH | LOCK_NB) == 0 {
            flock(fd, LOCK_UN)
            return false
        }  // nobody holds it
        return true
    }
    /// pid of the process holding the sampler lock, if one does (e.g. orphaned by a crashed app).
    private func lockHolder() -> Int32? {
        guard lockIsHeld(),
            let text = try? String(contentsOfFile: lockPath, encoding: .utf8),
            let pid = Int32(text.trimmingCharacters(in: .whitespacesAndNewlines)), pid > 1
        else { return nil }
        return pid
    }
    private func stopExternalSampler() {
        guard let pid = lockHolder() else { return }
        kill(pid, SIGTERM)
        DispatchQueue.main.asyncAfter(deadline: .now() + 5) { [weak self] in
            if let self, self.lockHolder() == pid { kill(pid, SIGKILL) }
        }
    }
    private func process(_ script: String, args: [String] = []) -> Process {
        let p = Process()
        p.executableURL = root.appendingPathComponent("venv/bin/python")
        p.arguments = [root.appendingPathComponent(script).path] + args
        p.currentDirectoryURL = root
        return p
    }
    private var restartAfterExit = false
    /// After sleep, Chromium's connection and any page load in flight are dead and the data is stale. A fresh sampler is the
    /// cleanest way back: it launches a new browser and takes a sample immediately.
    private func restartSamplerAfterWake() {
        lastWatchdogAction = Date()  // the CSV is stale after sleep; give the new sampler time before the watchdog judges it
        if let current = sampler, current.isRunning {
            restartAfterExit = true
            current.terminate()  // the termination handler starts the replacement
            DispatchQueue.main.asyncAfter(deadline: .now() + 8) { if current.isRunning { kill(current.processIdentifier, SIGKILL) } }
        } else {
            startSampler()
        }
        refresh()
    }
    func startSampler() {
        guard sampler?.isRunning != true else { return }
        guard !needsSetup else {
            samplerText = "Waiting for setup"
            return
        }
        // Someone else (a crashed app's orphan, a manual run) owns the lock: spawning would only log and exit 75 every tick.
        if lockIsHeld() {
            samplerText = "Using external sampler"
            return
        }
        let p = process("commute_sampler.py")
        let log = root.appendingPathComponent("commute_sampler.log")
        let oldLog = root.appendingPathComponent("commute_sampler.log.1")
        if let size = (try? FileManager.default.attributesOfItem(atPath: log.path))?[.size] as? Int, size > 1_000_000 {
            try? FileManager.default.removeItem(at: oldLog)
            try? FileManager.default.moveItem(at: log, to: oldLog)
        }
        if !FileManager.default.fileExists(atPath: log.path) { _ = FileManager.default.createFile(atPath: log.path, contents: nil) }
        do {
            let handle = try FileHandle(forWritingTo: log)
            try handle.seekToEnd()
            p.standardOutput = handle
            p.standardError = handle
            p.terminationHandler = { [weak self] process in
                Task { @MainActor in
                    guard let self, !self.stopping else { return }
                    self.samplerText =
                        process.terminationStatus == 75
                        ? "Another sampler is running" : "Sampler stopped (\(process.terminationStatus)). See log."
                    if self.sampler === process { self.sampler = nil }
                    if self.restartAfterExit {
                        self.restartAfterExit = false
                        self.startSampler()
                    }
                }
            }
            try p.run()
            sampler = p
            samplerText = "Sampling every 10 min"
            try handle.close()
        } catch { samplerText = "Sampler unavailable: \(error.localizedDescription)" }
    }
    func refresh() {
        guard !needsSetup else { return }
        guard !refreshing else { return }
        refreshing = true
        lastWatchdogAction = Date()  // a fresh sample is on its way; don't call the sampler stalled meanwhile
        startSampler()
        let p = process("commute_sampler.py", args: ["--request-refresh"])
        p.standardOutput = FileHandle.nullDevice
        p.standardError = FileHandle.nullDevice
        p.terminationHandler = { [weak self] process in
            Task { @MainActor in
                self?.refreshing = false
                if process.terminationStatus != 0 { self?.error = "Could not request a traffic refresh." }
                self?.load()
            }
        }
        do {
            try p.run()
            // A wedged requester would otherwise leave `refreshing` set forever, disabling the button for good.
            DispatchQueue.main.asyncAfter(deadline: .now() + 30) { [weak p] in
                if let p, p.isRunning { p.terminate() }
            }
        } catch {
            refreshing = false
            self.error = error.localizedDescription
        }
    }
    /// Newest modification time of the sample CSVs and the settings file; unchanged means a reload would show the same data.
    private func dataSignature() -> Date? {
        let manager = FileManager.default
        let names = ((try? manager.contentsOfDirectory(atPath: root.path)) ?? [])
            .filter { ($0.hasPrefix("commute_live") && $0.hasSuffix(".csv")) || $0 == "commute_settings.json" }
        return names.compactMap {
            (try? manager.attributesOfItem(atPath: root.appendingPathComponent($0).path))?[.modificationDate] as? Date
        }.max()
    }
    private func loadIfNeeded() {
        if dataSignature() != lastSignature || Date().timeIntervalSince(lastLoad) >= reloadInterval { load() }
    }
    func load() {
        needsSetup = !FileManager.default.fileExists(atPath: root.appendingPathComponent("commute_settings.json").path)
        if needsSetup {
            snapshot = nil
            error = nil
            return
        }
        guard !loading else {
            reloadPending = true
            return
        }
        loading = true
        lastLoad = Date()
        lastSignature = dataSignature()
        let revision = dataRevision
        reloadPending = false
        let p = process("commute_data.py")
        let timeout = dataTimeout
        DispatchQueue.global(qos: .utility).async { [weak self] in
            let result = Result { try runCapturingOutput(p, timeout: timeout) }
            Task { @MainActor in
                guard let self else { return }
                self.loading = false
                guard revision == self.dataRevision else {
                    self.load()
                    return
                }
                defer { if self.reloadPending { self.load() } }
                do {
                    let (data, status) = try result.get()
                    guard status == 0 else {
                        throw NSError(domain: "Commute", code: 1, userInfo: [NSLocalizedDescriptionKey: "Could not read commute data."])
                    }
                    let decoder = JSONDecoder()
                    decoder.keyDecodingStrategy = .convertFromSnakeCase
                    self.snapshot = try decoder.decode(Snapshot.self, from: data)
                    self.error = self.snapshot?.sourceError
                    self.resolveAddresses()
                    self.maybeNotify()
                    self.maybeNotifyHot()
                } catch { self.error = error.localizedDescription }
            }
        }
    }
    func settingsDidChange() {
        dataRevision += 1
        lastWatchdogAction = Date()  // the empty snapshot below must not look like a stalled sampler
        home = nil
        work = nil
        detectedDirection = nil
        addressKey = ""
        lastGeocodeFailure = nil
        snapshot = nil
        notifyEnabled = readNotifySetting()
        requestNotificationAuthorization()
        load()
        refresh()
    }
    // MARK: Launch at login
    var launchAtLoginEnabled: Bool { SMAppService.mainApp.status == .enabled }
    var launchAtLoginNeedsApproval: Bool { SMAppService.mainApp.status == .requiresApproval }
    func setLaunchAtLogin(_ enabled: Bool) throws {
        if enabled { try SMAppService.mainApp.register() } else { try SMAppService.mainApp.unregister() }
    }
    // MARK: Notifications
    /// `notify` from commute_settings.json; read directly so no Python runs on each tick.
    private func readNotifySetting() -> Bool {
        guard let data = try? Data(contentsOf: root.appendingPathComponent("commute_settings.json")),
            let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return true }
        return object["notify"] as? Bool ?? true
    }
    /// A numeric setting from commute_settings.json, same read path as `notify`.
    private func readNumberSetting(_ key: String) -> Double? {
        guard let data = try? Data(contentsOf: root.appendingPathComponent("commute_settings.json")),
            let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return nil }
        return object[key] as? Double
    }
    /// An accessory app counts as active while its Settings window is open; without this macOS would drop the banner.
    nonisolated func userNotificationCenter(
        _ center: UNUserNotificationCenter, willPresent notification: UNNotification,
        withCompletionHandler completionHandler: @escaping (UNNotificationPresentationOptions) -> Void
    ) {
        completionHandler([.banner, .sound])
    }
    private func requestNotificationAuthorization() {
        guard Bundle.main.bundleIdentifier != nil else { return }
        UNUserNotificationCenter.current().delegate = self
        guard notifyEnabled else { return }
        let center = UNUserNotificationCenter.current()
        center.getNotificationSettings { settings in
            if settings.authorizationStatus == .notDetermined { center.requestAuthorization(options: [.alert, .sound]) { _, _ in } }
        }
    }
    /// One "time to leave" alert per direction per day, once the suggested departure is within five minutes.
    /// Skipped when you're already at the destination, and on weekends and days off: those days are measured and
    /// forecast, but never alert.
    private func maybeNotify() {
        guard notifyEnabled, !stopping, Bundle.main.bundleIdentifier != nil, let direction,
            (snapshot?.dayStatus ?? "weekday") == "weekday", !isAtDestination(direction),
            let route = snapshot?.routes[direction], route.recommendation.status == "available",
            let departText = route.recommendation.departAt, let depart = parseDate(departText),
            depart <= Date().addingTimeInterval(5 * 60)
        else { return }
        let today = localDay(Date())
        var notified = UserDefaults.standard.dictionary(forKey: "notified") as? [String: String] ?? [:]
        guard notified[direction] != today else { return }
        notified[direction] = today
        UserDefaults.standard.set(notified, forKey: "notified")
        let recommendation = route.recommendation
        let content = UNMutableNotificationContent()
        let place = direction == "to_home" ? "home" : "work"
        content.title = recommendation.isLatest ? "Leave for \(place) by \(clockTime(depart))" : "Time to leave for \(place)"
        let minutes =
            (recommendation.expectedMinutes ?? recommendation.minutes).map { "About \(Int($0.rounded())) min drive" } ?? "Good time to go"
        content.body = selection == "auto" && detectedDirection != nil ? "\(minutes) · \(locationText)" : minutes
        content.sound = .default
        let request = UNNotificationRequest(identifier: "leave-\(direction)-\(today)", content: content, trigger: nil)
        let center = UNUserNotificationCenter.current()
        center.getNotificationSettings { settings in
            switch settings.authorizationStatus {
            case .authorized, .provisional: center.add(request)
            case .notDetermined:  // the permission prompt is still pending; try again on a later load
                Task { @MainActor in
                    var current = UserDefaults.standard.dictionary(forKey: "notified") as? [String: String] ?? [:]
                    if current[direction] == today {
                        current[direction] = nil
                        UserDefaults.standard.set(current, forKey: "notified")
                    }
                }
            default: break
            }
        }
    }
    /// One "traffic is running worse than this morning's forecast" alert per direction per day, fired while the
    /// suggested departure is still comfortably ahead — the point is leaving sooner than planned, so once the
    /// five-minute leave window starts the regular alert owns the notification instead.
    private func maybeNotifyHot() {
        let threshold = max(4, readNumberSetting("extra_drive_minutes") ?? 5)
        guard notifyEnabled, !stopping, Bundle.main.bundleIdentifier != nil, let direction,
            (snapshot?.dayStatus ?? "weekday") == "weekday", !isAtDestination(direction),
            let route = snapshot?.routes[direction], route.recommendation.status == "available",
            let offset = route.forecastOffsetMinutes, offset >= threshold,
            let departText = route.recommendation.departAt, let depart = parseDate(departText),
            depart > Date().addingTimeInterval(10 * 60)
        else { return }
        let today = localDay(Date())
        let key = "hot-\(direction)"
        var notified = UserDefaults.standard.dictionary(forKey: "notified") as? [String: String] ?? [:]
        guard notified[key] != today else { return }
        notified[key] = today
        UserDefaults.standard.set(notified, forKey: "notified")
        let place = direction == "to_home" ? "home" : "work"
        let content = UNMutableNotificationContent()
        content.title = "Heavier traffic than expected"
        var body = "~\(Int(offset.rounded())) min over this morning's forecast"
        if let latest = route.latest { body += " · now ~\(Int(latest.minutes)) min" }
        body += " — leave for \(place) around \(clockTime(depart))"
        content.body = body
        content.sound = .default
        let request = UNNotificationRequest(identifier: "\(key)-\(today)", content: content, trigger: nil)
        let center = UNUserNotificationCenter.current()
        center.getNotificationSettings { settings in
            switch settings.authorizationStatus {
            case .authorized, .provisional: center.add(request)
            case .notDetermined:  // the permission prompt is still pending; unmark so a later load can retry
                Task { @MainActor in
                    var current = UserDefaults.standard.dictionary(forKey: "notified") as? [String: String] ?? [:]
                    if current[key] == today {
                        current[key] = nil
                        UserDefaults.standard.set(current, forKey: "notified")
                    }
                }
            default: break
            }
        }
    }
    // MARK: Address lookup and location
    private var addressKey = ""
    private var resolving = false
    private func resolveAddresses() {
        guard let route = snapshot?.routes["to_work"] else { return }
        let key = route.origin + "\n" + route.destination
        if key != addressKey {
            home = nil
            work = nil
            detectedDirection = nil
            addressKey = key
            lastGeocodeFailure = nil
        }
        guard (home == nil || work == nil), !resolving else { return }
        if let failed = lastGeocodeFailure, Date().timeIntervalSince(failed) < 5 * 60 { return }
        resolving = true
        Task {
            do {
                let newHome = try await geocoder.geocodeAddressString(route.origin).first?.location
                let newWork = try await geocoder.geocodeAddressString(route.destination).first?.location
                if key == addressKey {
                    home = newHome
                    work = newWork
                    classifyLocation()
                    if newHome == nil || newWork == nil { lastGeocodeFailure = Date() }
                }
            } catch {
                if key == addressKey {
                    locationText = "Address lookup unavailable · choose a direction"
                    lastGeocodeFailure = Date()
                }
            }
            resolving = false
            if key != addressKey { resolveAddresses() }
        }
    }
    private let atPlaceRadius: CLLocationDistance = 600
    /// True when the last known location is already at the destination of `direction` (e.g. at work while heading to work).
    private func isAtDestination(_ direction: String) -> Bool {
        guard let location = lastLocation, let place = direction == "to_home" ? home : work else { return false }
        return location.distance(from: place) <= atPlaceRadius
    }
    private func classifyLocation() {
        guard let location = lastLocation, let home, let work else { return }
        let homeDistance = location.distance(from: home), workDistance = location.distance(from: work)
        if min(homeDistance, workDistance) <= atPlaceRadius {
            detectedDirection = homeDistance < workDistance ? "to_work" : "to_home"
            locationText = homeDistance < workDistance ? "At home · heading to work" : "At work · heading home"
        } else {
            detectedDirection = nil
            locationText = "Away from home and work · choose a direction"
        }
    }
    func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        guard let location = locations.last, location.horizontalAccuracy >= 0,
            location.horizontalAccuracy <= 500, abs(location.timestamp.timeIntervalSinceNow) < 300
        else { return }
        lastLocation = location
        classifyLocation()
    }
    func locationManager(_ manager: CLLocationManager, didFailWithError error: Error) {
        detectedDirection = nil
        locationText = "Location unavailable · choose a direction"
    }
    func locationManagerDidChangeAuthorization(_ manager: CLLocationManager) {
        if manager.authorizationStatus == .denied || manager.authorizationStatus == .restricted {
            detectedDirection = nil
            locationText = "Location permission off · choose a direction"
        }
    }
}
