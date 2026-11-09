import Foundation

private let localZone = TimeZone.current
private let localCalendar: Calendar = {
    var c = Calendar(identifier: .gregorian)
    c.timeZone = localZone
    return c
}()
private let plainISO = ISO8601DateFormatter()
private let fractionalISO: ISO8601DateFormatter = {
    let f = ISO8601DateFormatter()
    f.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
    return f
}()
private let clockFormatter: DateFormatter = {
    let f = DateFormatter()
    f.timeZone = localZone
    f.dateFormat = "h:mm a"
    return f
}()
private let dayFormatter: DateFormatter = {
    let f = DateFormatter()
    f.timeZone = localZone
    f.locale = Locale(identifier: "en_US_POSIX")
    f.dateFormat = "yyyy-MM-dd"
    return f
}()

struct Observation: Decodable, Identifiable {
    let observedAt: String
    let minutes: Double
    let miles: Double?
    let via: String
    let note: String
    let date: Date
    let minuteOfDay: Double
    var id: String { observedAt }
    private enum CodingKeys: String, CodingKey { case observedAt, minutes, miles, via, note }
    init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        observedAt = try values.decode(String.self, forKey: .observedAt)
        minutes = try values.decode(Double.self, forKey: .minutes)
        miles = try values.decodeIfPresent(Double.self, forKey: .miles)
        via = try values.decode(String.self, forKey: .via)
        note = try values.decode(String.self, forKey: .note)
        date = parseDate(observedAt) ?? .distantPast
        minuteOfDay = localMinute(date)
    }
}
struct HistoryPoint: Decodable, Identifiable {
    let minuteOfDay: Int
    let predicted: Double  // the model's expected drive time (older snapshots called it typical_minutes)
    let p10Minutes: Double
    let p90Minutes: Double
    let minMinutes: Double?  // only the sample-based baseline reports these
    let maxMinutes: Double?
    let sampleCount: Int?
    let dayCount: Int?
    let rangeLow: Int?  // the 80% range as whole minutes (the sampled drive times are whole minutes)
    let rangeHigh: Int?
    var id: Int { minuteOfDay }
    var typicalMinutes: Double { predicted }
    private enum CodingKeys: String, CodingKey {
        case minuteOfDay, predictedMinutes, typicalMinutes, p10Minutes, p90Minutes, minMinutes, maxMinutes, sampleCount, dayCount, rangeLow,
            rangeHigh
    }
    init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        minuteOfDay = try values.decode(Int.self, forKey: .minuteOfDay)
        if let value = try values.decodeIfPresent(Double.self, forKey: .predictedMinutes) {
            predicted = value
        } else {
            predicted = try values.decode(Double.self, forKey: .typicalMinutes)
        }
        p10Minutes = try values.decode(Double.self, forKey: .p10Minutes)
        p90Minutes = try values.decode(Double.self, forKey: .p90Minutes)
        minMinutes = try values.decodeIfPresent(Double.self, forKey: .minMinutes)
        maxMinutes = try values.decodeIfPresent(Double.self, forKey: .maxMinutes)
        sampleCount = try values.decodeIfPresent(Int.self, forKey: .sampleCount)
        dayCount = try values.decodeIfPresent(Int.self, forKey: .dayCount)
        rangeLow = try values.decodeIfPresent(Int.self, forKey: .rangeLow)
        rangeHigh = try values.decodeIfPresent(Int.self, forKey: .rangeHigh)
    }
}
/// How the model has actually done lately: average miss and how often the actual drive landed inside its range.
struct ModelReadout: Decodable {
    let maeMinutes: Double
    let hitRate: Double  // 0...1
    let days: Int
}
/// One scored day in the model's ledger: how the model did vs. the simple comparator curve.
struct ModelDay: Decodable, Identifiable {
    let date: String  // yyyy-MM-dd (local)
    let n: Int?  // forecast targets scored that day
    let mae: Double  // average miss of the model's expected drive time (minutes)
    let cov80: Double  // share of actual drives inside the model's 80% range (0...1)
    let is80: Double?  // interval score of the 80% band (lower is better)
    let cmpMae: Double?  // the same scores for the simple typical curve, when comparable
    let cmpIs80: Double?
    /// What the model leaned on that morning: phi/tau (the weekday share and the day-to-day swing) and the
    /// covariate coefficients — sib, prev, rain (min per mm), rain_mu, ve2 — plus n_days of evidence.
    let learned: [String: Double]?
    var id: String { date }
}
/// The thresholds a direction climbs through on the way to the "Predicted" label.
struct ModelGates: Decodable {
    let cal: Int  // scored days before the model's curve is shown at all
    let ready: Int  // scored days before the "Predicted" label can be earned
    let cov: Double  // target coverage of the prediction range (0.80)
}
struct RouteModel: Decodable {
    let label: String?  // "Predicted", "Typical (learning)", "Collecting data", ...
    let state: String?  // collecting | learning | calibrating | ready | fallback
    let days: Int?
    let intervalPercent: Int?
    let readout: ModelReadout?
    let note: String?
    let ledgerDays: Int?  // total days the model has been scored on
    let series: [ModelDay]?  // most recent scored days, oldest first
    let gates: ModelGates?
    let phi: Double?  // fitted share of day-to-day variance that is a weekday effect (0 = off)
    let tau: Double?  // the day-to-day swing the model expects before anything is known (minutes)
    let cov: [String: Double]?  // learned covariate state: sib, prev, rain (min/mm), rain_mu, rain_days, ve2, n
}
struct ForecastPoint: Decodable, Identifiable {
    let minuteOfDay: Int
    let minutes: Double
    var id: Int { minuteOfDay }
}
struct Recommendation: Decodable {
    let status: String
    let strategy: String?  // "earliest" (going home) or "latest" (going to work)
    let departAt: String?
    let minutes: Double?
    let expectedMinutes: Double?
    let sampleCount: Int  // distinct days behind the estimate
    let fastestMinutes: Double?
    let fastestDepartAt: String?
    let extraDriveMinutes: Double
    let forDate: String?  // set on a next-departure suggestion: the commuting day it is for (yyyy-MM-dd, local)
    var isLatest: Bool { strategy == "latest" }
    var isMiddle: Bool { strategy == "middle" }
    var strategyPhrase: String {
        isMiddle
            ? "The departure closest to the middle of your window that is within"
            : isLatest ? "The latest departure within" : "The earliest departure within"
    }
}
/// Precipitation on the forecast day, from the model's weather feed — only present on the learned path.
struct RainInfo: Decodable {
    let mm: Double
    let src: String?  // "obs" measured so far, "fcst" forecast
    let date: String  // the curve's ISO day (local)
}
struct Route: Decodable {
    let windowStart: Int
    let windowEnd: Int
    let curveDate: String?  // the day the curve and suggestion are for (yyyy-MM-dd, local): today while its window is ahead or open
    let origin: String
    let destination: String
    let url: String
    let latest: Observation?
    let today: [Observation]
    let todayAll: [Observation]?  // every sample today, any hour (the all-day graph); nil in older snapshots
    let history: [HistoryPoint]
    let base: [HistoryPoint]?  // the morning-of forecast (no today data): what the day was expected to look like
    let forecast: [ForecastPoint]?  // always empty now: the predicted curve already includes today's pace
    let forecastOffsetMinutes: Double?
    let rain: RainInfo?
    let model: RouteModel?
    let recommendation: Recommendation
    let nextDeparture: Recommendation?  // when today's window is over or it is a day off: the next commuting day's suggestion
    /// What to show: today's suggestion while there is one, otherwise the next day's.
    var shownRecommendation: Recommendation { recommendation.status != "available" ? (nextDeparture ?? recommendation) : recommendation }
    var forecastPoints: [ForecastPoint] { forecast ?? [] }  // only the simple curve has one; the Predicted curve already includes today's pace
    /// True when the curve drawn is the learned model's (calibrating/ready) rather than the simple typical curve.
    var showsModel: Bool { modelState == "calibrating" || modelState == "ready" }
    var modelLabel: String { model?.label ?? "Typical" }
    var modelState: String { model?.state ?? "ready" }
    var intervalPercent: Int { model?.intervalPercent ?? 80 }
}
struct Snapshot: Decodable {
    let generatedAt: String
    let timezone: String
    let sourceError: String?
    let skippedRows: Int
    let dayStatus: String?
    let dayOffReason: String?
    let routes: [String: Route]
}
private final class TimeoutFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var fired = false
    func fire() {
        lock.lock()
        fired = true
        lock.unlock()
    }
    var didFire: Bool {
        lock.lock()
        defer { lock.unlock() }
        return fired
    }
}
/// Runs `process` to completion and returns its stdout and exit status. A script still running after `timeout` seconds is
/// terminated (then killed), so one hung run can never wedge the caller. Blocks the calling thread.
func runCapturingOutput(_ process: Process, timeout: TimeInterval) throws -> (data: Data, status: Int32) {
    let pipe = Pipe()
    process.standardOutput = pipe
    process.standardError = FileHandle.nullDevice
    try process.run()
    let flag = TimeoutFlag()
    let watchdog = DispatchWorkItem {
        guard process.isRunning else { return }
        flag.fire()
        process.terminate()
        DispatchQueue.global().asyncAfter(deadline: .now() + 2) { if process.isRunning { kill(process.processIdentifier, SIGKILL) } }
    }
    DispatchQueue.global().asyncAfter(deadline: .now() + timeout, execute: watchdog)
    let data = pipe.fileHandleForReading.readDataToEndOfFile()
    process.waitUntilExit()
    watchdog.cancel()
    if flag.didFire { throw NSError(domain: "Commute", code: 2, userInfo: [NSLocalizedDescriptionKey: "Reading commute data timed out."]) }
    return (data, process.terminationStatus)
}
func parseDate(_ text: String) -> Date? {
    plainISO.date(from: text) ?? fractionalISO.date(from: text)
}
func clockTime(_ date: Date) -> String { clockFormatter.string(from: date) }
/// Minutes after midnight in local time, matching the sampler's windows.
func localMinute(_ date: Date) -> Double {
    Double(localCalendar.component(.hour, from: date) * 60 + localCalendar.component(.minute, from: date))
}
func localDay(_ date: Date) -> String { dayFormatter.string(from: date) }
/// A yyyy-MM-dd local day as a Date, for chart axes.
func dayDate(_ iso: String) -> Date? { dayFormatter.date(from: iso) }
/// "Tomorrow", or the weekday name, for a yyyy-MM-dd local date.
func dayLabel(_ iso: String) -> String {
    guard let date = dayFormatter.date(from: iso) else { return iso }
    let today = localCalendar.startOfDay(for: Date())
    let days = localCalendar.dateComponents([.day], from: today, to: localCalendar.startOfDay(for: date)).day ?? 0
    if days == 0 { return "Today" }
    if days == 1 { return "Tomorrow" }
    let f = DateFormatter()
    f.timeZone = localCalendar.timeZone
    f.locale = Locale(identifier: "en_US_POSIX")
    f.dateFormat = "EEEE"
    return f.string(from: date)
}
