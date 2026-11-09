import AppKit
import SwiftUI

struct CommuteSettings: Codable {
    var home: String
    var work: String
    var toWorkWaypoints: [String]
    var toHomeWaypoints: [String]
    var toWorkStart: String
    var toWorkEnd: String
    var toHomeStart: String
    var toHomeEnd: String
    var extraDriveMinutes: Double
    var notify = true
    var skipUsHolidays = true
    var skipDates: [String] = []
    var weatherLat: Double?
    var weatherLon: Double?

    private enum CodingKeys: String, CodingKey {
        case home, work, toWorkWaypoints, toHomeWaypoints, toWorkStart, toWorkEnd, toHomeStart, toHomeEnd, extraDriveMinutes, notify,
            skipUsHolidays, skipDates, weatherLat, weatherLon
    }
    // Settings written before days off and notifications existed lack those keys.
    init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        home = try values.decode(String.self, forKey: .home)
        work = try values.decode(String.self, forKey: .work)
        toWorkWaypoints = try values.decode([String].self, forKey: .toWorkWaypoints)
        toHomeWaypoints = try values.decode([String].self, forKey: .toHomeWaypoints)
        toWorkStart = try values.decode(String.self, forKey: .toWorkStart)
        toWorkEnd = try values.decode(String.self, forKey: .toWorkEnd)
        toHomeStart = try values.decode(String.self, forKey: .toHomeStart)
        toHomeEnd = try values.decode(String.self, forKey: .toHomeEnd)
        extraDriveMinutes = try values.decode(Double.self, forKey: .extraDriveMinutes)
        notify = try values.decodeIfPresent(Bool.self, forKey: .notify) ?? true
        skipUsHolidays = try values.decodeIfPresent(Bool.self, forKey: .skipUsHolidays) ?? true
        skipDates = try values.decodeIfPresent([String].self, forKey: .skipDates) ?? []
        weatherLat = try values.decodeIfPresent(Double.self, forKey: .weatherLat)
        weatherLon = try values.decodeIfPresent(Double.self, forKey: .weatherLon)
    }
    static func read(root: URL) throws -> Self {
        let process = Process()
        process.executableURL = root.appendingPathComponent("venv/bin/python")
        process.arguments = [root.appendingPathComponent("commute_config.py").path]
        let (data, status) = try runCapturingOutput(process, timeout: 15)  // runs on the main thread: never wait forever
        guard status == 0 else {
            throw NSError(
                domain: "Settings", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Could not read settings. Check commute_settings.json for invalid values."])
        }
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(Self.self, from: data)
    }
    func save(root: URL) throws {
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(self).write(to: root.appendingPathComponent("commute_settings.json"), options: .atomic)
    }
}

struct SettingsView: View {
    @ObservedObject var model: CommuteModel
    /// Settings supplied up front (preview rendering has no window, so onAppear never runs); nil loads from disk when shown.
    init(model: CommuteModel, preloaded: CommuteSettings? = nil) {
        self.model = model
        self.preloaded = preloaded != nil
        _draft = State(initialValue: preloaded)
        _workWaypoints = State(initialValue: preloaded?.toWorkWaypoints.joined(separator: "\n") ?? "")
        _homeWaypoints = State(initialValue: preloaded?.toHomeWaypoints.joined(separator: "\n") ?? "")
        _skipDatesText = State(initialValue: preloaded?.skipDates.joined(separator: ", ") ?? "")
    }
    private let preloaded: Bool
    // @State written out by hand: the Command Line Tools here lack the SwiftUIMacros plugin that expands @State.
    private var _draft = State<CommuteSettings?>(initialValue: nil)
    private var draft: CommuteSettings? {
        get { _draft.wrappedValue }
        nonmutating set { _draft.wrappedValue = newValue }
    }
    private var _workWaypoints = State(initialValue: "")
    private var workWaypoints: String {
        get { _workWaypoints.wrappedValue }
        nonmutating set { _workWaypoints.wrappedValue = newValue }
    }
    private var _homeWaypoints = State(initialValue: "")
    private var homeWaypoints: String {
        get { _homeWaypoints.wrappedValue }
        nonmutating set { _homeWaypoints.wrappedValue = newValue }
    }
    private var _skipDatesText = State(initialValue: "")
    private var skipDatesText: String {
        get { _skipDatesText.wrappedValue }
        nonmutating set { _skipDatesText.wrappedValue = newValue }
    }
    private var _launchAtLogin = State(initialValue: false)
    private var launchAtLogin: Bool {
        get { _launchAtLogin.wrappedValue }
        nonmutating set { _launchAtLogin.wrappedValue = newValue }
    }
    private var _message = State<String?>(initialValue: nil)
    private var message: String? {
        get { _message.wrappedValue }
        nonmutating set { _message.wrappedValue = newValue }
    }
    private var _saved = State(initialValue: false)
    private var saved: Bool {
        get { _saved.wrappedValue }
        nonmutating set { _saved.wrappedValue = newValue }
    }
    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            Text("Commute settings").font(.title2.weight(.semibold))
            if draft != nil {
                VStack(alignment: .leading, spacing: 10) {
                    Text("Locations").font(.headline)
                    HStack {
                        Text("Home").frame(width: 45, alignment: .leading)
                        TextField("Home address", text: field(\.home)).textFieldStyle(.roundedBorder)
                    }
                    HStack {
                        Text("Work").frame(width: 45, alignment: .leading)
                        TextField("Work address", text: field(\.work)).textFieldStyle(.roundedBorder)
                    }
                }
                VStack(alignment: .leading, spacing: 10) {
                    Text("Departure windows").font(.headline)
                    windowRow("To work", start: field(\.toWorkStart), end: field(\.toWorkEnd))
                    windowRow("To home", start: field(\.toHomeStart), end: field(\.toHomeEnd))
                    Text(
                        "Weekdays · local time · 24-hour format. Traffic is sampled 24/7; these ranges only control which samples the Predicted graph and the best-departure estimates use. The Today graph shows every sample."
                    )
                    .font(.caption).foregroundStyle(.secondary)
                }
                VStack(alignment: .leading, spacing: 8) {
                    Text("Driving allowance").font(.headline)
                    Stepper(
                        value: Binding(
                            get: { draft?.extraDriveMinutes ?? 5 },
                            set: {
                                draft?.extraDriveMinutes = $0
                                saved = false
                                message = nil
                            }), in: 0...30, step: 1
                    ) {
                        let minutes = draft?.extraDriveMinutes ?? 5
                        Text("\(minutes.formatted()) extra driving minute\(minutes == 1 ? "" : "s")")
                    }
                    Text(
                        "Going home: leave as early as possible while within this much of the fastest remaining drive. Going to work: leave near the middle of your window, just before or after the rush; a bigger allowance favors the middle more. Zero waits for the fastest option."
                    )
                    .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                }
                VStack(alignment: .leading, spacing: 8) {
                    Text("Preferred routes").font(.headline)
                    Text(
                        "Enter intermediate streets, intersections, or addresses in driving order, one per line. Leave empty for a direct route."
                    )
                    .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                    HStack(alignment: .top, spacing: 14) {
                        waypointEditor("To work", text: _workWaypoints.projectedValue)
                        waypointEditor("To home", text: _homeWaypoints.projectedValue)
                    }
                }
                VStack(alignment: .leading, spacing: 8) {
                    Text("Days off").font(.headline)
                    Toggle("Treat US federal holidays as days off", isOn: flag(\.skipUsHolidays))
                    HStack {
                        Text("Other days off").frame(width: 100, alignment: .leading)
                        TextField(
                            "YYYY-MM-DD, comma separated",
                            text: Binding(
                                get: { skipDatesText },
                                set: {
                                    skipDatesText = $0
                                    saved = false
                                    message = nil
                                })
                        )
                        .textFieldStyle(.roundedBorder)
                    }
                    Text(
                        "Days off are still measured, but forecast from their own history instead of mixing with normal days. They never send notifications."
                    )
                    .font(.caption).foregroundStyle(.secondary)
                }
                VStack(alignment: .leading, spacing: 8) {
                    Text("General").font(.headline)
                    Toggle("Notify me when it's time to leave", isOn: flag(\.notify))
                    Toggle("Launch at login", isOn: Binding(get: { launchAtLogin }, set: { setLaunchAtLogin($0) }))
                    Button("Show data in Finder") { showData() }
                }
                Text("Changing locations or waypoints starts separate route history. Your previous samples are retained.")
                    .font(.caption).foregroundStyle(.secondary)
                HStack {
                    if let message {
                        Text(message).font(.caption).foregroundStyle(saved ? Color.secondary : Color.orange).fixedSize(
                            horizontal: false, vertical: true)
                    }
                    Spacer()
                    Button("Save") { save() }.keyboardShortcut("s", modifiers: .command).buttonStyle(.borderedProminent)
                }
            } else if let message {
                Text(message).foregroundStyle(.orange)
            } else {
                Text("Loading…").foregroundStyle(.secondary)
            }
        }
        .padding(24).frame(width: 540).background(Color(nsColor: .windowBackgroundColor))
        .onAppear {
            launchAtLogin = model.launchAtLoginEnabled
            if !preloaded { load() }
        }
    }
    private func field(_ key: WritableKeyPath<CommuteSettings, String>) -> Binding<String> {
        Binding(
            get: { draft?[keyPath: key] ?? "" },
            set: {
                draft?[keyPath: key] = $0
                saved = false
                message = nil
            })
    }
    private func flag(_ key: WritableKeyPath<CommuteSettings, Bool>) -> Binding<Bool> {
        Binding(
            get: { draft?[keyPath: key] ?? true },
            set: {
                draft?[keyPath: key] = $0
                saved = false
                message = nil
            })
    }
    private func windowRow(_ title: String, start: Binding<String>, end: Binding<String>) -> some View {
        HStack {
            Text(title).frame(width: 70, alignment: .leading)
            TextField("06:00", text: start).frame(width: 72)
            Text("to").foregroundStyle(.secondary)
            TextField("12:00", text: end).frame(width: 72)
            Spacer()
        }.textFieldStyle(.roundedBorder)
    }
    private func waypointEditor(_ title: String, text: Binding<String>) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title).font(.subheadline.weight(.medium))
            TextEditor(text: text).font(.system(size: 12)).frame(height: 115)
                .padding(4).overlay(RoundedRectangle(cornerRadius: 6).stroke(Color.secondary.opacity(0.3)))
        }
    }
    private func load() {
        do {
            let settings = try CommuteSettings.read(root: model.root)
            draft = settings
            workWaypoints = settings.toWorkWaypoints.joined(separator: "\n")
            homeWaypoints = settings.toHomeWaypoints.joined(separator: "\n")
            skipDatesText = settings.skipDates.joined(separator: ", ")
        } catch { message = error.localizedDescription }
    }
    private func setLaunchAtLogin(_ enabled: Bool) {
        do {
            try model.setLaunchAtLogin(enabled)
            launchAtLogin = model.launchAtLoginEnabled
            saved = model.launchAtLoginNeedsApproval ? false : saved
            if enabled && model.launchAtLoginNeedsApproval { message = "Approve Commute Bar in System Settings → General → Login Items." }
        } catch {
            launchAtLogin = model.launchAtLoginEnabled
            saved = false
            message = "Could not change launch at login: \(error.localizedDescription)"
        }
    }
    private func showData() {
        let files = ((try? FileManager.default.contentsOfDirectory(atPath: model.root.path)) ?? [])
            .filter { $0.hasPrefix("commute_live") && $0.hasSuffix(".csv") }.sorted()
            .map { model.root.appendingPathComponent($0) }
        if files.isEmpty { NSWorkspace.shared.open(model.root) } else { NSWorkspace.shared.activateFileViewerSelecting(files) }
    }
    /// YYYY-MM-DD tokens that are real calendar dates, sorted and de-duplicated; nil names the first bad token.
    private func parseDays(_ text: String) -> (days: [String], invalid: String?) {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone(identifier: "UTC")
        formatter.dateFormat = "yyyy-MM-dd"
        formatter.isLenient = false
        var days = Set<String>()
        for token in text.components(separatedBy: CharacterSet(charactersIn: ", \n\t;")).filter({ !$0.isEmpty }) {
            guard token.range(of: "^\\d{4}-\\d{2}-\\d{2}$", options: .regularExpression) != nil,
                let date = formatter.date(from: token), formatter.string(from: date) == token
            else { return ([], token) }
            days.insert(token)
        }
        return (days.sorted(), nil)
    }
    private func save() {
        guard var settings = draft else { return }
        settings.home = settings.home.trimmingCharacters(in: .whitespacesAndNewlines)
        settings.work = settings.work.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !settings.home.isEmpty, !settings.work.isEmpty else {
            saved = false
            message = "Enter both home and work addresses."
            return
        }
        guard settings.extraDriveMinutes.isFinite, (0...30).contains(settings.extraDriveMinutes) else {
            saved = false
            message = "Extra driving allowance must be between 0 and 30 minutes."
            return
        }
        func minute(_ text: String) -> Int? {
            guard text.range(of: "^(?:[01][0-9]|2[0-3]):[0-5][0-9]$", options: .regularExpression) != nil else { return nil }
            let parts = text.split(separator: ":", omittingEmptySubsequences: false)
            guard parts.count == 2, parts[0].count == 2, parts[1].count == 2,
                let h = Int(parts[0]), let m = Int(parts[1]), (0...23).contains(h), (0...59).contains(m)
            else { return nil }
            return h * 60 + m
        }
        for (start, end) in [(settings.toWorkStart, settings.toWorkEnd), (settings.toHomeStart, settings.toHomeEnd)] {
            guard let a = minute(start), let b = minute(end), a < b else {
                saved = false
                message = "Use HH:mm times with the end later than the start (same day)."
                return
            }
        }
        let parsedDays = parseDays(skipDatesText)
        if let invalid = parsedDays.invalid {
            saved = false
            message = "“\(invalid)” is not a valid day off. Use YYYY-MM-DD dates separated by commas."
            return
        }
        settings.skipDates = parsedDays.days
        func waypoints(_ text: String) -> [String] {
            text.components(separatedBy: .newlines).map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }.filter { !$0.isEmpty }
        }
        settings.toWorkWaypoints = waypoints(workWaypoints)
        settings.toHomeWaypoints = waypoints(homeWaypoints)
        do {
            try settings.save(root: model.root)
            draft = settings
            skipDatesText = settings.skipDates.joined(separator: ", ")
            model.settingsDidChange()
            saved = true
            message = "Saved. Fresh traffic requested."
        } catch {
            saved = false
            message = error.localizedDescription
        }
    }
}
