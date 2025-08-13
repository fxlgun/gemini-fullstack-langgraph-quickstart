import { useState, useEffect } from "react";
import { auth, RecaptchaVerifier, signInWithPhoneNumber } from "./firebase";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectTrigger, SelectContent, SelectItem, SelectValue } from "@/components/ui/select";
import countryList from "./countries.json";
import { Loader2 } from "lucide-react";
import { Toaster, toast } from "sonner";

export default function PhoneLogin() {
  const [selectedCountry, setSelectedCountry] = useState({
    name: "India",
    flag: "https://upload.wikimedia.org/wikipedia/en/4/41/Flag_of_India.svg",
    number: "+91",
  });
  const [phone, setPhone] = useState("");
  const [otp, setOtp] = useState("");
  const [confirmation, setConfirmation] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [resendTimer, setResendTimer] = useState(0);

  // Timer countdown for resend
  useEffect(() => {
    let timer: NodeJS.Timeout;
    if (resendTimer > 0) {
      timer = setTimeout(() => setResendTimer(resendTimer - 1), 1000);
    }
    return () => clearTimeout(timer);
  }, [resendTimer]);

  const sendOtp = async () => {
    try {
      setLoading(true);
      (window as any).recaptchaVerifier = new RecaptchaVerifier(auth, "recaptcha-container", {
        size: "invisible",
      });

      const fullPhone = `${selectedCountry.number}${phone}`;
      const confirmationResult = await signInWithPhoneNumber(
        auth,
        fullPhone,
        (window as any).recaptchaVerifier
      );

      setConfirmation(confirmationResult);
      toast.success("OTP sent successfully!");
      setResendTimer(30);
      toast.success("OTP sent to your phone");
    } catch (err) {
      toast.error("OTP sent to your phone");
      console.error(err);
      toast.error("Error sending OTP");
    } finally {
      setLoading(false);
    }
  };

  const verifyOtp = async () => {
    try {
      setLoading(true);
      await confirmation.confirm(otp);
      toast.success("Phone number verified successfully!");
    } catch (err) {
      console.error(err);
      toast.error("Invalid OTP");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 p-4">
      <Toaster richColors position="top-center"/>
      <Card className="w-full max-w-md shadow-lg">
        <CardHeader>
          <CardTitle className="text-center text-lg font-semibold">
            Sign in with Phone
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Country Selector */}
          <Select
            onValueChange={(val) => {
              const country = countryList.find((c) => c.number === val);
              if (country) setSelectedCountry(country);
            }}
            defaultValue={selectedCountry.number}
          >
            <SelectTrigger>
              <SelectValue>
                <div className="flex items-center gap-2">
                  <img src={selectedCountry.flag} alt={selectedCountry.name} className="w-5 h-5 rounded-sm" />
                  <span>{selectedCountry.name} ({selectedCountry.number})</span>
                </div>
              </SelectValue>
            </SelectTrigger>
            <SelectContent>
              {countryList.map((country) => (
                <SelectItem key={country.flag} value={country.number}>
                  <div className="flex items-center gap-2">
                    <img src={country.flag} alt={country.name} className="w-5 h-5 rounded-sm" />
                    {country.name} ({country.number})
                  </div>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          {/* Phone Input */}
          <div className="flex gap-2">
            <span className="px-3 py-2 bg-gray-100 border rounded-md text-gray-700">{selectedCountry.number}</span>
            <Input
              placeholder="Enter phone number"
              value={phone}
              onChange={(e) => setPhone(e.target.value)}
            />
          </div>

          {/* OTP Input */}
          {confirmation && (
            <Input
              placeholder="Enter OTP"
              value={otp}
              onChange={(e) => setOtp(e.target.value)}
            />
          )}

          {/* Buttons */}
          {!confirmation ? (
            <Button className="w-full" onClick={sendOtp} disabled={loading}>
              {loading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : "Send OTP"}
            </Button>
          ) : (
            <>
              <Button className="w-full" onClick={verifyOtp} disabled={loading}>
                {loading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : "Verify OTP"}
              </Button>
              <Button
                variant="outline"
                className="w-full"
                onClick={sendOtp}
                disabled={resendTimer > 0 || loading}
              >
                {resendTimer > 0 ? `Resend OTP in ${resendTimer}s` : "Resend OTP"}
              </Button>
            </>
          )}

          {/* Recaptcha */}
          <div id="recaptcha-container"></div>
        </CardContent>
      </Card>
    </div>
  );
}
